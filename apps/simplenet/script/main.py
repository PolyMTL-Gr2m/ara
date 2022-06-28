import logging
import argparse
from ModelParser import ModelParser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--model', help='input model', required=True)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = parse_args()
    model = ModelParser(args['model'])
    # model.print_model_graph()
    model.gen_code()