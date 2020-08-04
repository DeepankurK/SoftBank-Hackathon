# This file is to create sentence_vector of each encrypted date by considering sll the data pints available of the text data for a specific date and getting a 1x300 vector for each date byttaking its sum.
#importing
import tarfile
import csv
import io
import pandas as pd

frames = []
for i in range(241):
    print(i)
    df = pd.read_csv('./competition_data/text_features/chunk_{}.csv.gz'.format(i), compression='gzip', header=0,    sep=',', quotechar='"', error_bad_lines=False)
    #grouping by id and taking its mean
    new_df = df.groupby('id').mean()
    print(new_df.shape[0])
    frames.append(new_df)
#concating and storing
result = pd.concat(frames)
result.to_csv("sentence_vectors.csv", sep=',')