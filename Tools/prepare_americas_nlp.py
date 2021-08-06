"""
Script to prepare the data from the AmericasNLP 2021 repository, with the names
of the files to prepare being read from the command line.

The command line arguments are the file path for each of the AmericasNLP files
the user wants to prepare.

The data preparation for each file removes the lines that contain urls or
copyright symbols. The remaining lines are then output to a file with the same
name as the input file (the output file overwrites the input file for each file
being prepared). The script also produces a combined dev file and combined train
file, which will have all the prepared lines from each of the dev files/each of
the train files given as command line arguments (and assumes files will have the
same name that they do in the AmericasNLP repository, for example, 'dev.cni') 
"""
import sys
from typing import List


def prepare(filename: str) -> List[str]:
    """
    Prepare the data from AmericasNLP 2021 by removing the lines that contain
    urls and/or copyright symbols

    Args:
        filename: The path of the file (from the AmericasNLP dataset) to prepare
    Returns:
        A list of strings, which are the prepared lines from the given data file 
    """
    # get lines from file
    lines = [line.strip() for line in open(filename, 'r')]
    lines[:] = [x for x in lines if x]

    # remove lines with urls and copyright symbols
    prepared = [
        line for line in lines if 'www.' not in line and '©' not in line
    ]

    return prepared


def write_to_file(filename: str, prepared_data: List[str]):
    """
    Write lines from a list of strings to a file. Add a newline after each
    string
    
    Args:
        filename: The name of the file to write to
        prepared_data: The list of strings to write to the file 
    """
    f = open(filename, 'w')
    for line in prepared_data:
        s = line + '\n'
        f.write(s)
    f.close()


def main():
    """
    Main function of the program. Creates a list of files to prepare from the
    command line arguments. For each file, it calls prepare() and
    write_to_file(). This function also sets up a combined dev file with all of
    the prepared data from all of the dev files from the command line arguments.
    It is assumed that the files will have the same name that they do in the
    AmericasNLP repository, for example 'dev.cni'. This function sets up a
    similar file for training data as well.
    """
    # get list of files
    list_of_files = sys.argv[1:]
    dev_files = [filename for filename in list_of_files if 'dev' in filename]
    dev_files.sort()
    train_files = [
        filename for filename in list_of_files if 'train' in filename
    ]
    train_files.sort()

    # to store lists of prepared lines for the combined dev file and the
    # combined train file
    dev_combined_prepared = []
    train_combined_prepared = []

    # prepare dev data and write to files
    for filename in dev_files:
        prepared = prepare(filename)
        write_to_file(filename, prepared)
        for line in prepared:
            dev_combined_prepared.append(line)

    # write to combined dev file
    write_to_file('dev.anlp', dev_combined_prepared)

    # prepare train data and write to files
    for filename in train_files:
        prepared = prepare(filename)
        write_to_file(filename, prepared)
        for line in prepared:
            train_combined_prepared.append(line)

    # write to combined train file
    write_to_file('train.anlp', train_combined_prepared)


if __name__ == "__main__":
    main()
