import json
import argparse
import os
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(os.path.join("./exps", args.model + ".json"))
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--model', type=str, default='dsease',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
