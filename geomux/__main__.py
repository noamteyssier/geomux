import argparse
from .utils import read_table

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input table to assign")
    args = parser.parse_args()
    return args


def main_cli():
    args = get_args()
    frame = read_table(args.input)
    print(frame)

