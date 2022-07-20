import requests
from pprint import pprint
import json 

URL = "http://127.0.0.1:8000/"


def get_config(algorithm="maxcut", architecture="QAOA"):
    config_final = {}
    config_all = {
        "config":
            {
            "QAOA":{
                "general":{ # General parameters in the QAOA algorithm
                    "func_find_par": { # Name of the function in the QAOA algorithm
                        "method": "COBYLA", # Method to use in optimization
                        "tolerance": 0.001, # Tolerance of the optimization
                        "options": {"maxiter": 100000} # Options of the optimization
                    },
                    "func_find_best_parameters": {
                        "pool_size": 10, # Number of virtual processors to use
                        "iter_init": 100, # Number of solutions to generate for every P-value
                        "best_number": 15, # Number of best solutions to keep for the next P-value
                        "p_start": 3, # Starting P-value
                        "p_end": 10, # Ending P-value
                        "use_next_params": True, # Use the parameters learned in the previous P-value
                        "beta_corr_thr": 0.9, # Threshold for the beta correction to be accepted as good solution
                        "gamma_corr_thr": 0.9 # Threshold for the gamma correction to be accepted as good solution
                    }
                },
                "jsp": {
                    "max_time": 4, # Maximum time that the solution can take
                    "problem_version": "optimization" # Version of the problem to solve (optimization or decizion)
                },
                "maxcut": {
                    "n": -1, # Number of vertices in random graph (if -1, data taken from input) 
                    #"a": 1
                }
            },
            
            "D-Wave": {
                "jsp":{
                    "mode": "sim_pyqubo",
                    "num_reads": 1000,
                    "weight_one_hot": 3,
                    "weight_precedence": 1,
                    "weight_share":2
                }
            }
        }
    }
    config_final["config"] = {}
    if architecture == "QAOA":
        config_final["config"]["general"] = config_all["config"]["QAOA"]["general"]
        config_final["config"][algorithm] = config_all["config"]["QAOA"][algorithm]

    elif architecture == "D-Wave":
        config_final["config"]["D-Wave"] = {algorithm: config_all["config"]["D-Wave"][algorithm]}

    return config_final

def add_data_to_config(config, data, algorithm):
    config["data"] = data
    config["algorithm"] = algorithm
    return config

def visualize(data, algorithm):
    pass

def send_request_to_solve(config, token):
    headers = {'Authorization': f'Token {token}'}
    id = requests.post(f"{URL}solve/", json=config, headers=headers).json()
    return id

def get_solution(id, token):
    headers = {'Authorization': f'Token {token}'}
    response = requests.get(f"{URL}result/{id}", headers=headers).json()
    return response

def get_solution_sync(id, token):
    result = {}
    result["status"] = None
    while result["status"] != "Solved":
        result = get_solution(id, token)
    return result

def maxcut(V, E, architecture = "QAOA", token = "2cc03f168032ae77fb28e5a2229a89acdf7ee89b", get_result_async = True):
    config = get_config(algorithm=maxcut.__name__, architecture=architecture)
    data = {"Vertices": V, "Edges": E}
    algorithm = "maxcut"
    visualize(data, algorithm)
    config_with_data = add_data_to_config(config, data, algorithm)

    id = send_request_to_solve(config_with_data, token)
    print(f'Solving problem {algorithm} with id: {id}')

    if not get_result_async:
        response = get_solution_sync(id, token)
        return response


