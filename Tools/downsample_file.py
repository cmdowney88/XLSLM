import argparse
import random

def main():
    """
    Script to down-sample a file to a new number of lines, choosing those lines
    randomly without replacement. Random seed defaults to 1
    """
    parser = argparse.ArgumentParser(description="MBart Translation Script")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--new_num_lines', type=int, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)

    original_lines = [
        line.strip() for line in open(args.input_file, 'r') if line != ''
    ]

    new_lines = random.sample(original_lines, args.new_num_lines)

    with open(args.output_file, 'w+') as fout:
        for line in new_lines:
            print(line, file=fout)

if __name__ == "__main__":
    main()
