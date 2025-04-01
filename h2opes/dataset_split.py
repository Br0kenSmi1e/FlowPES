import pandas as pd
from random import random

def split(datafile, train=0.7, valid=0.2):
    data = pd.read_csv(datafile)
    nitems = len(data)
    train_idx = []
    valid_idx = []
    test_idx = []
    for idx in range(nitems):
        u = random()
        if u < train:
            train_idx.append(idx)
        elif u < train + valid:
            valid_idx.append(idx)
        else:
            test_idx.append(idx)

    file_name, file_ext = datafile.split(".")

    train_data = {'theta': [data['theta'][idx] for idx in train_idx], 'r1': [data['r1'][idx] for idx in train_idx], 'r2': [data['r2'][idx] for idx in train_idx], 'energy': [data['energy'][idx] for idx in train_idx]}
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(file_name + "_train." + file_ext)

    valid_data = {'theta': [data['theta'][idx] for idx in valid_idx], 'r1': [data['r1'][idx] for idx in valid_idx], 'r2': [data['r2'][idx] for idx in valid_idx], 'energy': [data['energy'][idx] for idx in valid_idx]}
    valid_df = pd.DataFrame(valid_data)
    valid_df.to_csv(file_name + "_valid." + file_ext)

    test_data = {'theta': [data['theta'][idx] for idx in test_idx], 'r1': [data['r1'][idx] for idx in test_idx], 'r2': [data['r2'][idx] for idx in test_idx], 'energy': [data['energy'][idx] for idx in test_idx]}
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(file_name + "_test." + file_ext)

split("h2opes_analytic.txt")