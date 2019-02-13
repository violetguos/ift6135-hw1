"""generates parameters to json file like args.json"""


import json
import argparse
import pprint

def setup_parser(arguments, title):

    parser = argparse.ArgumentParser(description=title,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, val in arguments.items():
        parser.add_argument('-%s' % key,
                            type=eval(val["type"]),
                            help=val["help"],
                            default=val["default"])

    return parser


def read_params(parser):

    parameters = vars(parser.parse_args())
    return parameters

def get_parameters(data, title=None):

    #data = load_json(data_path)
    parser = setup_parser(data, title)
    parameters = read_params(parser)

    return parameters


def load_json(data_path):
    with open(data_path, "r") as fp:
        json_dict = json.load(fp)
    return json_dict

if __name__ == "__main__":
    params = get_parameters()
    pprint.pprint(params)