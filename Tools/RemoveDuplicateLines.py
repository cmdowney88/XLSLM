"""
Script to remove any duplicate lines between AmericasNLP files and KannEtAl2018
files for the same language. If a line exists in both the AmericasNLP data and
KannEtAl2018 data, it is removed from KannEtAl2018 and stays in AmericasNLP. The
script is only meant for one language at a time

Expected command line args: <program_name> <AmericasNLP_train_file> \
    <AmericasNLP_dev_file> <KannEtAl2018_file> 
"""
import re
import sys
from typing import List


def remove(anlp_lines: List[str], kea_lines: List[str]) -> List[str]:
    """
    Remove duplicate lines between AmericasNLP data and Kann et al. 2018 data
    from the Kann et al. 2018 data. Returns the remaining lines for the Kann et
    al. 2018 data
    
    Args:
        anlp_lines: A list of the lines in the AmericasNLP files
        kea_lines: A list of the lines in the Kann et al. file.
    Returns:
        A list of strings, which are the remaining lines from the Kann et al.
        file after removing duplicates
    """
    # to store anlp lines with no whitespace or punctuation
    stripped_anlp_lines = []

    # remove whitespace and punctuation for anlp lines for comparison with kae
    # lines
    for line in anlp_lines:
        stripped_line = no_whitespace_or_punct(line)
        stripped_anlp_lines.append(stripped_line)

    # to store the final lines from Kann et al. 2018 remaining after removing
    # any duplicates
    final_kea_lines = []

    # remove whitespace and punctuation for Kann et al. lines and compare with
    # the lines from AmericasNLP with whitespace and punctuation removed. If not
    # a duplicate line, add Kann et al. line to list of final lines
    for line in kea_lines:
        orig_line = line
        stripped_line = no_whitespace_or_punct(line)
        if stripped_line not in stripped_anlp_lines:
            final_kea_lines.append(orig_line)

    return final_kea_lines


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
    lines between AmericasNLP and Kann et al. 2018 from Kann et al. 2018 with
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

    # combine lines from both AmericasNLP files into one list
    anlp_lines = anlp_train_lines + anlp_dev_lines

    # get lines from Kann et al. 2018 file
    kea_lines = [line.strip() for line in open(kea_file, 'r')]
    kea_lines[:] = [x for x in kea_lines if x]

    # Remove duplicate lines and write remaining lines to file
    final_kea_lines = remove(anlp_lines, kea_lines)
    write_to_file(kea_file[:-4] + '_removed' + kea_file[-4:], final_kea_lines)


if __name__ == "__main__":
    main()
