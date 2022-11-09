#ExactCover
from collections import defaultdict
import numpy as np
from qat.core import Observable, Term


#QAOA
from pprint import pprint
import random
import json
import scipy
import matplotlib.pyplot as plt
from math import ceil
from p_tqdm import p_map
from scipy.stats import pearsonr
from qat.vsolve.ansatz import AnsatzFactory
from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin



class Algorithm:
    def __init__(self, config, beta_bound=np.pi * 1, gamma_bound=np.pi * 2):

        self.beta_bound = beta_bound
        self.gamma_bound = gamma_bound

        self.solutions = []
        self.config = config
        self.progress_p = 0
        self.progress_i = 0
        self.p_range = [0, 0]
        self.best_iter_solution = -1
        self.iter_loop = 0
        self.jobs_solution = -1
        self.best_tape_graph = []
        self.not_enough_calculation = 0
        self.pool_size = self.config["parameters"]["find_best_parameters_qaoa"]["pool_size"]
        self.iter_init = self.config["parameters"]["find_best_parameters_qaoa"]["iter_init"]
        self.best_number = self.config["parameters"]["find_best_parameters_qaoa"]["best_number"]
        self.p_start = self.config["parameters"]["find_best_parameters_qaoa"]["p_start"]
        self.use_educated_guess = self.config["parameters"]["find_best_parameters_qaoa"]["use_next_params"]

        if self.use_educated_guess:
            self.p_end = self.config["parameters"]["find_best_parameters_qaoa"]["p_end"] + 1
        else:
            self.p_end = self.p_start + 1

    def match_params_to_job(self, x, job, p):
        params = []
        for var in job.get_variables():
            idx = int(var.split('_')[-1].replace('{', '').replace('}', ''))
            if 'beta' in var:
                params.append(x[idx])
            elif 'gamma' in var:
                params.append(x[p + idx])
        return params

    def find_par(self, hamiltonian, i, p, educated_guess_params, use_educated_guess):
        x = None
        np.random.seed(random.randint(0, 2 ** 23 - 1))
        x = np.random.rand(2 * p)
        x[:p] = x[:p] * self.beta_bound
        x[p:] = x[p:] * self.gamma_bound
        circuit = AnsatzFactory.qaoa_circuit(hamiltonian, p)
        job = circuit.to_job(observable=hamiltonian)
        qpu = get_default_qpu()

        init_params = self.match_params_to_job(x, job, p)
        lb = np.zeros(len(x))
        ub = np.ones(len(x))
        ub[:p] *= self.beta_bound
        ub[p:] *= self.gamma_bound
        constraint = scipy.optimize.LinearConstraint(np.eye(len(x)), lb, ub)
        stack = ScipyMinimizePlugin(method=self.config["parameters"]["find_qaoa"]["method"],
                                    tol=self.config["parameters"]["find_qaoa"]["tolerance"], x0=init_params,
                                    constraints=constraint,
                                    options={"maxiter": self.config["parameters"]["find_qaoa"]["options"]}
                                    ) | qpu
        result = stack.submit(job)
        optimal_job = circuit.to_job()
        optimal_job = optimal_job(
            **{var: val for var, val in zip(job.get_variables(), json.loads(result.meta_data['parameters']))})
        result_optimal = qpu.submit(optimal_job)

        counts = {x.state.bitstring: x.probability for x in result_optimal}
        energy, energies = self.compute_energy(counts)
        var_map_dict = eval(result.meta_data['parameter_map'])
        items = [[x[0].split('_'), x[1]] for x in list(var_map_dict.items())]
        sorted_params = sorted(items, key=lambda i: [i[0][0], int(
            i[0][1].replace('{', '').replace('}', ''))])
        params = [x[1] for x in sorted_params]

        print("", end="")
        return (energy, params, counts, energies)



    # def create_heatmap(self, args):
    #     x, hamiltonian, p, p_i, side_length, verbose = args

    #     def beta_function(beta, side_length):
    #         return - 2 * \
    #                np.pi + 4 * np.pi * beta / (side_length - 1)

    #     def gamma_function(gamma, side_length):
    #         return - 2 * \
    #                np.pi + 4 * np.pi * gamma / (side_length - 1)

    #     energies = np.zeros((side_length, side_length))
    #     ansatz_with_cnots = AnsatzFactory.qaoa_circuit(hamiltonian, p)
    #     qpu = get_default_qpu()
    #     job = ansatz_with_cnots.to_job()

    #     for beta in range(0, side_length):
    #         for gamma in range(0, side_length):
    #             if verbose:
    #                 print(beta, gamma)

    #             x[p_i] = beta_function(beta, side_length)
    #             x[p_i + p] = gamma_function(gamma, side_length)
    #             init_params = self.match_params_to_job(x, job, p)

    #             optimal_job = job(
    #                 **{var: val for var, val in zip(job.get_variables(), init_params)})
    #             result_optimal = qpu.submit(optimal_job)
    #             counts = {
    #                 x.state.bitstring: x.probability for x in result_optimal}
    #             if type(self).__name__ == "JSP":
    #                 energy, _ = self.compute_maxcut_energy(
    #                     counts, self.revese_dict_map, self.model)
    #             elif type(self).__name__ == "MAXCUT":
    #                 energy, _ = self.compute_maxcut_energy(counts, self.G)
    #             elif type(self).__name__ == "EXACTCOVER":
    #                 energy, _ = self.compute_maxcut_energy(counts, self.routes)
    #             energies[beta, gamma] = energy

    #     return energies

    # def create_heatmap_from_best(self, params, p, p_i, hamiltonian, side_length=50):
    #     x = np.copy(params)

    #     def beta_function(beta, side_length): return np.pi * \
    #                                                  beta / (side_length - 1)

    #     def gamma_function(gamma, side_length): return 2 * \
    #                                                    np.pi * gamma / (side_length - 1)

    #     energies = self.create_heatmap(
    #         x, hamiltonian, p, p_i, side_length, beta_function, gamma_function)

    #     lowest_indices = np.unravel_index(np.argmin(energies), energies.shape)
    #     lowest_beta = np.pi * lowest_indices[0] / (side_length - 1)
    #     lowest_gamma = np.pi * 2 * lowest_indices[1] / (side_length - 1)

    #     return energies, lowest_beta, lowest_gamma

    # def find_new_params(self, solution, p, hamiltonian):
    #     new_params = np.zeros(2 * p)
    #     _, params, _, _ = solution
    #     for i, (beta, gamma) in enumerate(zip(params[:p], params[p:])):
    #         if beta > 0 and beta < np.pi and gamma > 0 and gamma < 2 * np.pi:
    #             new_params[i] = beta
    #             new_params[i + p] = gamma
    #         else:
    #             heatmap, lowest_beta, lowest_gamma = self.create_heatmap_from_best(
    #                 params, p, i, hamiltonian)
    #             new_params[i] = lowest_beta
    #             new_params[i + p] = lowest_gamma
    #     return new_params

    # def fix_tape(self, args):
    #     solution, i, hamiltonian, p = args
    #     new_params = self.find_new_params(solution, p, hamiltonian)
    #     use_educated_guess = True
    #     new_solution = self.find(
    #         [hamiltonian, i, p, new_params, use_educated_guess])
    #     # print(new_solution)
    #     return new_solution

    def range_energies(self, x, interval_count):
        przedzial = ceil(len(x) / interval_count)
        #   print(przedzial)
        interval = []
        for i in range(ceil(len(x) / przedzial)):
            if ((i + 1) * przedzial + 1 <= len(x)):
                interval.append(sorted(list(set([x[i * przedzial], x[(i + 1) * przedzial]]))))
            else:
                if (x[i * przedzial] != x[-1]):
                    interval.append(sorted(list(set([x[i * przedzial], x[-1]]))))
                else:
                    if (interval[-1][-1] != x[-1]):
                        interval.append([x[-1], '∞'])
        if (interval[-1][-1] != '∞'):
            interval.append([interval[-1][-1], '∞'])
        return interval

    def range_probability(self, y, interval_count):
        przedzial = ceil(len(y) / interval_count)
        interval = []
        for i in range(ceil(len(y) / przedzial)):
            if ((i + 1) * przedzial <= len(y)):
                interval.append(self.sum_it(y[i * przedzial:(i + 1) * przedzial]))
            else:
                interval.append(self.sum_it(y[i * przedzial:]))
        return interval

    def sum_it(self, lista):
        return round(sum(lista), 3)
    
    def check_number_of_output_jobs(self, output):
        output_no_blanks_sum = 0
        for job in output:
            if job[0][0]=="_":
                continue
            output_no_blanks_sum += 1
        jobs_sum = 0
        for x in self.jobs:
            if isinstance(self.jobs[x], list):
                jobs_sum += len(self.jobs[x])
        if output_no_blanks_sum == jobs_sum:
            return True
        return False

    def find_best_parameters(self):

        self.progress_p = self.p_start
        self.p_range = [self.p_start, self.p_end]
        educated_guess_params = None


        if self.best_number > self.iter_init:
            self.best_number = self.iter_init

        iter_loop = self.iter_init
        hamiltonian = self.make_hamiltonian()

        for p in range(self.p_start, self.p_end):
            print("\n\n")
            if self.p_start + 1 != self.p_end:
                print("Iteration for p = ", p)

            self.iter_loop = iter_loop
            tape = p_map(find_run, [[hamiltonian, i, p, educated_guess_params, self.use_educated_guess, self] for i in
                                    range(iter_loop)], num_cpus=self.pool_size)

            self.progress_p = p + 1
            if educated_guess_params is None:
                tape = sorted(tape, key=lambda x: x[0])

            best_tape = tape
            if p > 1:
                correlations = [(pearsonr(np.arange(1, p + 1), x[1][:p])[0],
                                 pearsonr(np.arange(1, p + 1), x[1][p:])[0]) for x in tape]
                best_tape = []
                for t, c in zip(tape, correlations):
                    if c[0] > self.config["parameters"]["find_best_parameters2_qaoa"]["beta_corr_thr"] and c[1] > \
                            self.config["parameters"]["find_best_parameters2_qaoa"]["gamma_corr_thr"]:
                        best_tape.append(t)

            best_tape = best_tape[:self.best_number]
            try:
                self.solutions.append((p, best_tape[0][2], best_tape[0][0]))
            except:
                print("Nothing good enought to add, probably not enought iteretion")


            iter_loop = int(len(best_tape))
            if len(best_tape) > 0:
                self.best_iter_solution = max(
                    best_tape[0][2], key=best_tape[0][2].get)
                x, y = zip(*best_tape[0][3].items())
                zipped_lists = zip(x, y)
                sorted_pairs = sorted(zipped_lists, reverse=False)
                tuples = zip(*sorted_pairs)
                xs, ys = [tuple for tuple in tuples]
                yss = self.range_probability(ys, 5)
                xss = self.range_energies(xs, 5)

                interval = []
                # print('X:',xss,'\nY:',yss, flush=True)
                if (len(yss) + 1 == len(xss)):
                    yss.append(0)
                for i in range(len(xss)):
                    interval.append({"probability": yss[i], "energy": (str(xss[i])).replace("'", "").replace("]", ")")})
                self.best_tape_graph.append(interval)



            self.best_iter_solution = ''.join('1' if x == '0' else '0' for x in self.best_iter_solution)

        best_solution = ""
        self.progress_p = -1
        try:
            best_solution = max(
                self.solutions[-1][1], key=self.solutions[-1][1].get)
        except:
            pass
        print("\n")
        print("graph_data:", self.best_tape_graph, '\n')
        print("Najlepsze rozwiazanie to:", ''.join('1' if x == '0' else '0' for x in best_solution))
        return (''.join('1' if x == '0' else '0' for x in best_solution))


class EXACTCOVER(Algorithm):
    def __init__(self, routes,  config, beta_bound=np.pi * 2, gamma_bound = np.pi * 2):
        Algorithm.__init__(self, config, beta_bound, gamma_bound)
        self.routes = routes
        self.Jrr_dict = -1
        self.hr_dict = -1

    def Jrr(self, route1, route2):
        s = len(set(route1).intersection(set(route2)))
        return s / 2

    def hr(self, route1, routes):
        i_sum = 0
        for r in routes:
            i_sum += len(set(r).intersection(set(route1)))
        s = i_sum - len(route1) * 2
        return s / 2

    def calculate_jrr_hr(self):
        Jrr_dict = dict()
        indices = np.triu_indices(len(self.routes), 1)
        for i1, i2 in zip(indices[0], indices[1]):
            Jrr_dict[(i1, i2)] = self.Jrr(self.routes[i1], self.routes[i2])

        hr_dict = dict()
        for i in range(len(self.routes)):
            hr_dict[i] = self.hr(self.routes[i], self.routes)

        return Jrr_dict, hr_dict


    def make_hamiltonian(self):
        line_obs = Observable(len(self.routes))
        self.Jrr_dict, self.hr_dict = self.calculate_jrr_hr()
        for i in self.Jrr_dict:
            # print(i)
            line_obs.add_term(Term(self.Jrr_dict[i], "ZZ", [i[0], i[1]]))

        for i in self.hr_dict:
            # print(i)
            line_obs.add_term(Term(self.hr_dict[i], "Z", [i]))
        # print(line_obs)
        return line_obs

    def obj(self, x):
        spin = list(map(lambda x: 2*x - 1, list(map(int, x))))
     
        s1 = 0
        indices = np.triu_indices(len(x), 1)
        for i1, i2 in zip(indices[0], indices[1]):
            partial = spin[i1] * spin[i2] * self.Jrr_dict[(i1, i2)]
            s1 += partial
     
        s2 = 0
        for i in range(len(self.routes)):
            s2 += self.hr_dict[i] * spin[i]
     
        return s1 + s2

    def compute_energy(self, counts):
        energy = 0
        energies = defaultdict(int)
        total_counts = 0
        for meas, meas_count in counts.items():
            obj_for_meas = self.obj(meas)
            energies[obj_for_meas] += meas_count
            energy += obj_for_meas * meas_count
            total_counts += meas_count
        return energy / total_counts, energies


def find_run(args):
    hamiltonian, i, p, educated_guess_params, use_educated_guess, executing_class = args
    return (executing_class.find_par(hamiltonian, i, p, educated_guess_params, use_educated_guess))