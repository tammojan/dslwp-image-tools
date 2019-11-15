#!/usr/bin/env python

import numpy as np
import pandas as pd
from argparse import ArgumentParser

def select_tags(databasefile, tags):
    """
    Select a subset of DSLWP-B images according to their tags

    Args:
        databasefile (str): Name of database file
        tags (List[str]): tags, case insensitive, that should be in there.
                          prepend with ^ to exclude

    Returns:
        List[str]: filenames 
    """
    df = pd.read_csv(databasefile, delimiter='\t', index_col=1)

    select_col = df['Filename'].str.contains('jpeg') # Just True
    for tag in tags:
        if tag[0] == '^':
            select_col = select_col & ~df['Tags'].str.contains(tag[1:], case=False)
        else:
            select_col = select_col & df['Tags'].str.contains(tag, case=False)

    return df[select_col]["Filename"].tolist()

if __name__ == "__main__":
    parser = ArgumentParser(description="Select a subset of DSLWP-B images")
    parser.add_argument("databasefile", help="Database file")
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()

    filenames = select_tags(args.databasefile, args.tags)

    for filename in filenames:
        print(filename)
