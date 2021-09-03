"""
Script to prepare the K'iche' data from Global Classroom, with the paths of the
files to prepare being read from the command line arguments.

Expected order of command line arguments: <program_name> <train_data_file> \
    <dev_data_file> <test_data_file>

The data preparation for each file replaces '>' with ' ' for the segmented data
and removes any empty lines from both the segmented and unsegmented data. The
remaining lines from the segmented data are then output to a file and the
remaining lines from the unsegmented data are output to a separate file. This is
done for the train file, dev file, and test file.
"""
import sys
from typing import List, Tuple

import pandas as pd


def prepare(filename: str) -> Tuple[List[str], List[str]]:
    """
    Prepare the K'iche' data from the Global Classroom repository by replacing
    '>' with ' ' for the segmented data, and removing any empty lines from both
    the segmented and unsegmented data

    Args:
        filename: The path of the file to prepare. Should be a .tsv file.
    Returns:
        Two lists of strings, which contain the prepared segmented and
        unsegmented lines from the given data file
    """
    # open tsv file
    df = pd.read_csv(filename, names=["k'iche'", "segmented_k'iche'"], sep='\t')

    # prepare data
    segmented_data = [
        x.replace('>', ' ')
        for x in df["segmented_k'iche'"] if isinstance(x, str)
    ]
    unsegmented_data = [x for x in df["k'iche'"] if isinstance(x, str)]

    return segmented_data, unsegmented_data


def write_to_file(filename: str, prepared_data: List[str]):
    """
    Write lines from a list of strings to a file. Add a newline after each
    string
    
    Args:
        filename: The path of the file to write to
        prepared_data: The list of strings to write to the file
    """
    f = open(filename, 'w')
    for line in prepared_data:
        s = line + '\n'
        f.write(s)
    f.close()


def main():
    """
    Main function of the program. Gets the names of the files that contain the
    training data, dev data, test data from the command line arguments. For each
    data file, it prepares the segmented and unsegmented data, writes the
    segmented data to a file, and writes the unsegmented data to a file.
    
    Expected order of command line arguments: <program_name> <train_data_file> \
        <dev_data_file> <test_data_file> 
    """
    # get data files
    train_filename = sys.argv[1]
    dev_filename = sys.argv[2]
    test_filename = sys.argv[3]

    # quc is ISO 639-3 for K'iche'

    # For train file, prepare segmented and unsegmented data, and output
    # prepared data to files
    segmented_data, unsegmented_data = prepare(train_filename)
    write_to_file("segmented_train.quc", segmented_data)
    write_to_file("unsegmented_train.quc", unsegmented_data)

    # For dev file, prepare segmented and unsegmented data, and output prepared
    # data to files
    segmented_data, unsegmented_data = prepare(dev_filename)
    write_to_file("segmented_dev.quc", segmented_data)
    write_to_file("unsegmented_dev.quc", unsegmented_data)

    # For test file, prepare segmented and unsegmented data, and output prepared
    # data to files
    segmented_data, unsegmented_data = prepare(test_filename)
    write_to_file("segmented_test.quc", segmented_data)
    write_to_file("unsegmented_test.quc", unsegmented_data)


if __name__ == "__main__":
    main()
