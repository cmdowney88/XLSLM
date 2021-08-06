"""
Script to prepare the data from Kann et al. (2018), with the path of the Excel file being
read from the command line.
The command line argument is the path of the file to prepare.
The data preparation for each column removes lines that only say 'FALTA', replaces '-' with
' ', and makes commas and question marks separate tokens from the tokens they previously 
attached to.
The remaining lines from each column/language are output to a file for each column/language.
"""
import sys
from typing import List

import pandas as pd

def prepare(
    dataframe: str,
    column_name: str
) -> List[str]:
    """
    Prepare the data from Kann el al. 2018 Excel spreadsheet by removing lines that say only
    'FALTA', replacing '-' with ' ', and separating commas and question marks from the ends of
    tokens so that commas and question marks are their own separate tokens.
    Returns a list of strings, which are the remaining lines from the column given as an argument
    in the dataframe given as an argument.
    Args:
        dataframe: The name of the dataframe that holds the data
        column_name: The name of the column to prepare
    Returns:
        A list of strings, which are the prepared lines from the given column in the given
        dataframe
    """
    prepared = [x for x in dataframe[column_name] if isinstance(x,str) and x != 'FALTA']
    prepared[:] = [x.replace('-',' ').replace(',',' , ').replace('?',' ? ') for x in prepared]
    prepared[:] = [' '.join(x.split()) for x in prepared]

    return prepared


def write_to_file(
    filename: str,
    prepared_data: List[str]
):
    """
    Write lines from a list of strings to a file. Add a newline after each string.
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
    Main function of the program. Gets the file to prepare from the command line arguments.
    For each language/column, prepares the data with prepare(). For each language/column, writes 
    the prepared data to a file with write_to_file(). 
    """
    #get filename and read contents into pandas dataframe
    filename = sys.argv[1]
    sheet = 'Hoja 1'
    df = pd.read_excel(io = filename, sheet_name = sheet)

    #prepare data for each language/column (besides Spanish)
    prepared_purepecha = prepare(df, 'purepecha')
    prepared_wixarika = prepare(df, 'wix')
    prepared_yorem_nokki = prepare(df, 'yorem nokki')
    prepared_mexicanero = prepare(df, 'mexicanero')
    prepared_nahuatl = prepare(df, 'nahuatl')

    #write prepared data to files
    #ISO 639-3:
    #purepecha: pua, wixarika: hch, yorem nokki: mfy, mexicanero: azd, nahuatl: nci
    write_to_file('dev.pua', prepared_purepecha)
    write_to_file('dev.hch', prepared_wixarika)
    write_to_file('dev.mfy', prepared_yorem_nokki)
    write_to_file('dev.azd', prepared_mexicanero)
    write_to_file('dev.nci', prepared_nahuatl)


if __name__ == "__main__":
    main()
