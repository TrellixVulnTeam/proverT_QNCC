# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# saving relative data paths in dictionary

data_paths = {
    'A':{'indiv':{'train':'A_indiv_train.csv', 'test': 'A_indiv_test.csv'},
                    'hhold': {'train':'A_hhold_train.csv', 'test': 'A_hhold_test.csv'}},
    'B': {'indiv':{'train':'B_indiv_train.csv', 'test': 'B_indiv_test.csv'},
                    'hhold': {'train':'B_hhold_train.csv', 'test': 'B_hhold_test.csv'}},
    'C': {'indiv':{'train':'C_indiv_train.csv', 'test': 'C_indiv_test.csv'},
                    'hhold': {'train':'C_hhold_train.csv', 'test': 'C_hhold_test.csv'}}}


# loading data
# country A
a_i_train = pd.read_csv(data_paths['A']['indiv']['train'], index_col='id')
a_i_test = pd.read_csv(data_paths['A']['indiv']['test'], index_col='id')
a_h_train = pd.read_csv(data_paths['A']['hhold']['train'], index_col='id')
a_h_test = pd.read_csv(data_paths['A']['hhold']['test'], index_col='id')

# country B
b_i_train = pd.read_csv(data_paths['A']['indiv']['train'], index_col='id')
b_i_test = pd.read_csv(data_paths['A']['indiv']['test'], index_col='id')
b_h_train = pd.read_csv(data_paths['A']['hhold']['train'], index_col='id')
b_h_test = pd.read_csv(data_paths['A']['hhold']['test'], index_col='id')

# country C
c_i_train = pd.read_csv(data_paths['A']['indiv']['train'], index_col='id')
c_i_test = pd.read_csv(data_paths['A']['indiv']['test'], index_col='id')
c_h_train = pd.read_csv(data_paths['A']['hhold']['train'], index_col='id')
c_h_test = pd.read_csv(data_paths['A']['hhold']['test'], index_col='id')

# I will build 3 models for the three different countries. Starting at country A.

# To standardize, we will use the function provided by drivendata.org
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtract mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df


def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))


    df = standardize(df)
    print("After standardization {}".format(df.shape))

    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))


    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})

    df.fillna(0, inplace=True)

    return df

# standardizing all data for country A
aX_i_train = pre_process_data(a_i_train.drop('poor', axis = 1))
aY_i_train = np.ravel(a_i_train.poor) # converting from pandas series to np array

aX_h_train = pre_process_data(a_h_train.drop('poor', axis = 1))
aY_h_train = np.ravel(a_h_train.poor) # converting from pandas series to np array

aX_i_test = pre_process_data(a_i_test)
aX_h_test = pre_process_data(a_h_test)

# TODO
# combine hh and individual level data: functions!
# build rf model including cross validation: functions!
# repeat for all countries
# make submission
