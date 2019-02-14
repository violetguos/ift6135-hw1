"""generates a json of random params"""

from src.algo.hyperparam_gen import ParamGenerator
import json
import argparse
import sys
from src.utils.parser_helper import get_parameters, load_json

def run(args):
    pg = ParamGenerator(args.seed) # seed not used
    eta = pg.learningRate()
    h1, h2 = pg.hiddenUnit()
    data = { "h1":
                 {"type" : "int",
                  "help": "hidden",
                "default": h1
                  },
            "h2":
                {"type": "int",
                 "help": "hidden",
                 "default": h2},
             "learning_rate":{
                 "type": "float",
                 "help": "eta learning rate",
                 "default": eta
                }

            }

    return data




def main(argv):
    hyperparam_dict = {}
    hyperparam_dict['hyperparam'] = []

    parser = argparse.ArgumentParser(description='MLP with numpy')
    parser.add_argument('--seed', type=int, default=10, help='seed not used')
    args = parser.parse_args(argv)

    for i in range(5):
        hyperparam_dict['hyperparam'].append(run(args))
    with open("args3.json", "w") as data_file:
        json.dump(hyperparam_dict, data_file, indent=4)


if __name__ == "__main__":
    main(sys.argv[1:])

    config = load_json("args3.json")

    for hyper_parameter_set in config["hyperparam"]:
        p = get_parameters(hyper_parameter_set)
        #pprint.pprint(params)
        print(p)
