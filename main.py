import json
import argparse
import os
from trainer import train

'''
Installation with Conda:

conda create -n psrd_ease python=3.10
conda activate psrd_ease
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pip
pip install -r requirements.txt

RUN ON IMAGENET-R:
python main.py --model psrd_ease_inr

RUN ON IMAGENET-A:
python main.py --model psrd_ease_ina

To change device and hyperparameters:
navigate to /exps/psrd_ease_inr.json or /exps/psrd_ease_ina.json
'''

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
