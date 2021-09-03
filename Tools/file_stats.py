"""
Script to get stats on the data files in the XLSLM repository.
Command line arguments are the paths of the data files to get the stats for.
For each file, the script produces stats for the total number of lines, total
number of tokens, number of unique tokens, total number of characters, number of
characters in unique tokens, total number of unique characters, average length
of all tokens, and average length of unique tokens.
Each of the data files will have its stats output to a separate file. There will
also be a combined stats file with the stats for all of the files combined into
one stats file, where the stats are separated by the original data file names.
"""
import os
import sys
from typing import Dict, List


def get_stats(filename: str) -> Dict[str, str]:
    """
    Get statistics for a language data file
    
    For the given file, finds the total number of lines, total number of tokens,
    number of unique tokens, total number of characters, number of characters in
    unique tokens, total number of unique characters, average length of all
    tokens, and average length of unique tokens.

    Args:
        filename: The path of the file to find stats for
    Returns:
        A dict of key, value pairs that are both strings. The keys are the
        labels for the stats. The values are the string representations of the
        stats.
    """
    # dictionary to store stats on data
    stats_dict = {'Filename': filename}

    # open file and get lines
    lines = [line.strip().lower() for line in open(filename, 'r')]
    lines[:] = [x for x in lines if x]

    # find how many lines are in file
    num_lines = len(lines)
    stats_dict['Number of lines'] = str(num_lines)

    # find total number of tokens
    total_tokens_list = []
    for line in lines:
        tokens = line.split()
        for token in tokens:
            total_tokens_list.append(token)

    total_tokens_num = len(total_tokens_list)
    stats_dict['Total number of tokens'] = str(total_tokens_num)

    # find total number of unique tokens
    unique_tokens_list = list(set(total_tokens_list))
    unique_tokens_num = len(unique_tokens_list)
    stats_dict['Number of unique tokens'] = str(unique_tokens_num)

    # find total number of characters
    total_character_num_of_all_tokens = 0
    for token in total_tokens_list:
        total_character_num_of_all_tokens += len(token)
    stats_dict['Total number of characters in file'] = str(
        total_character_num_of_all_tokens
    )

    # fine total number of characters in unique tokens
    total_character_num_of_unique_tokens = 0
    for token in unique_tokens_list:
        total_character_num_of_unique_tokens += len(token)
    stats_dict['Total number of characters in unique tokens'] = str(
        total_character_num_of_unique_tokens
    )

    # find unique characters and total number of unique characters
    # total_chars_list = [char for token in total_tokens_list for char in token]
    total_chars_list = []
    for token in total_tokens_list:
        for char in token:
            total_chars_list.append(char)
    total_chars_list[:] = [char.lower() for char in total_chars_list]

    unique_chars_list = list(set(total_chars_list))
    unique_chars_num = len(unique_chars_list)
    stats_dict['Total number of unique characters'] = str(unique_chars_num)

    # find average length of all tokens
    total, counter = 0, 0
    for token in total_tokens_list:
        total += len(token)
        counter += 1
    avg_token_length = round(total / counter, 2)
    stats_dict['Average token length of all tokens in file'] = str(
        avg_token_length
    )

    # find average length of unique tokens
    total, counter = 0, 0
    for token in unique_tokens_list:
        total += len(token)
        counter += 1
    avg_unique_token_length = round(total / counter, 2)
    stats_dict['Average token length of all unique tokens in file'] = str(
        avg_unique_token_length
    )

    return stats_dict


def write_to_file(filename: str, stats_dict: Dict[str, str]):
    """
    Write lines to a file from a dictionary of string keys and string values.
    
    Args:
        filename: The name of the file to write to
        stats_dict: The dictionary of string keys and values to write to the
            file
    """
    f = open(filename, 'w')
    for k, v in stats_dict.items():
        s = str(k) + ': ' + str(v) + '\n'
        f.write(s)
    f.close()


def write_all_to_file(filename: str, list_of_stats_dicts: List[Dict[str, str]]):
    """
    Write lines to a file from a list of dictionaires with string keys and
    string values. For each dictionary in the list, prints each key, value pair
    to a line.

    Args:
        filename: The name of the file to write to
        list_of_stats_dicts: The list of dictionaries with string keys and
            values to write to the file
    """
    # overwrite old output file if it exists
    f = open(filename, 'w')
    f.close()

    # print to output file
    for stats_dict in list_of_stats_dicts:
        f = open(filename, 'a')
        for k, v in stats_dict.items():
            s = str(k) + ': ' + str(v) + '\n'
            f.write(s)
        f.write('\n')
        f.close()


def main():
    """
    Main function of the program. Gets the names of files to find stats for from
    the command line arguments. For each file, it finds the stats with
    get_stats() and writes them to a file with write_to_file(). Also writes
    stats from all files to combined stats file with write_all_to_file(). 
    """
    # get names of files to find stats for
    list_of_files = sys.argv[1:]
    list_of_files.sort(key=lambda x: os.path.splitext(x)[-1])

    # list to store all of the stats dicts to output to combined file
    list_of_stats = []

    # get stats for each file and write stats to file
    for filename in list_of_files:
        stats_dict = get_stats(filename)
        write_to_file(f"{filename}.stats", stats_dict)
        list_of_stats.append(stats_dict)

    # write to combined stats file
    write_all_to_file('combined_stats.txt', list_of_stats)


if __name__ == "__main__":
    main()
