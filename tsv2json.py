#!/usr/bin/env python

import pandas as pd

df = pd.read_csv("dslwp-image-database.tsv", sep="\t", index_col="ImageID")
df.to_json("dslwp-image-database.json", orient="index")
