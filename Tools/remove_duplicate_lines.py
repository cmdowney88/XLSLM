"""
Script to remove any duplicate lines between AmericasNLP files and KannEtAl2018
files for the same language. If a line exists in both the AmericasNLP data and
KannEtAl2018 data, it is removed from AmericasNLP and stays in KannEtAl. The
script is only meant for one language at a time

Expected command line args: <program_name> <AmericasNLP_train_file> \
    <AmericasNLP_dev_file> <KannEtAl2018_file> 
"""
import os
import re
import sys
from typing import List, Tuple


def remove(
    anlp_train_lines: List[str], anlp_dev_lines: List[str], kea_lines: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Remove duplicate lines between AmericasNLP data and Kann et al. 2018 data
    from the AmericasNLP data. Returns the remaining lines for the AmericasNLP
    data
    
    Args:
        anlp_train_lines: A list of the lines in the AmericasNLP train file
        anlp_dev_lines: A list of the lines in the AmericasNLP dev file
        kea_lines: A list of the lines in the Kann et al. file.
    Returns:
        A pair of lists of strings, which are the remaining lines from the
        AmericasNLP training and dev files after removing duplicates
    """
    # remove whitespace and punctuation for kea lines for comparison with anlp
    # lines
    stripped_kea_lines = [no_whitespace_or_punct(line) for line in kea_lines]

    # remove whitespace and punctuation for the AmericasNLP lines and compare
    # with the lines from Kann et al. with whitespace and punctuation removed.
    # Keep in AmericasNLP only if not a duplicate from Kann et al.
    final_anlp_train_lines = [
        line for line in anlp_train_lines
        if no_whitespace_or_punct(line) not in stripped_kea_lines
    ]
    final_anlp_dev_lines = [
        line for line in anlp_dev_lines
        if no_whitespace_or_punct(line) not in stripped_kea_lines
    ]

    return final_anlp_train_lines, final_anlp_dev_lines


def no_whitespace_or_punct(line: str) -> str:
    """
    Remove whitespace and punctuation from string

    Args:
        line: String to remove whitespace and punctuation from
    Returns:
        A string with the whitespace and punctuation removed
    """
    # remove whitespace
    newstr = line.replace(' ', '')

    # remove punctuation besides "'" and "+"
    non_phonetic_punc = r"[.?!,*%$=\\/[\]()\";]*"
    newstr = re.sub(non_phonetic_punc, '', newstr)

    return newstr


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
    Main function of program. Gets files from command lines. Removes duplicate
    lines between AmericasNLP and Kann et al. 2018 from AmericasNLP with
    remove() and writes the remaining lines to a file with write_to_file()
    """
    # Get file paths from command line
    anlp_train_file = sys.argv[1]
    anlp_dev_file = sys.argv[2]
    kea_file = sys.argv[3]

    # get lines from AmericasNLP files
    anlp_train_lines = [line.strip() for line in open(anlp_train_file, 'r')]
    anlp_train_lines[:] = [x for x in anlp_train_lines if x]

    anlp_dev_lines = [line.strip() for line in open(anlp_dev_file, 'r')]
    anlp_dev_lines[:] = [x for x in anlp_dev_lines if x]

    # get lines from Kann et al. 2018 file
    kea_lines = [line.strip() for line in open(kea_file, 'r')]
    kea_lines[:] = [x for x in kea_lines if x]

    # Remove duplicate lines and write remaining lines to file
    final_anlp_train_lines, final_anlp_dev_lines = remove(
        anlp_train_lines, anlp_dev_lines, kea_lines
    )
    base, extension = os.path.splitext(anlp_train_file)
    write_to_file(base + '_removed' + extension, final_anlp_train_lines)
    base, extension = os.path.splitext(anlp_dev_file)
    write_to_file(base + '_removed' + extension, final_anlp_dev_lines)


if __name__ == "__main__":
    main()
