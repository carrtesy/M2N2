"""
    Anoshift data preprocessing code
    https://github.com/bit-ml/AnoShift/blob/main/baselines_OOD_setup/load_anoshift.py
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import os, sys
import gc

train_years = [2006, 2007, 2008, 2009, 2010]
test_years = [2011, 2012, 2013, 2014, 2015]
years = train_years + test_years


def load(anoshift_db_path, year, use_valid=False):
    if use_valid:
        df = pd.read_parquet(os.path.join(anoshift_db_path, f'subset/{year}_subset_valid.parquet'))
    else:
        df = pd.read_parquet(os.path.join(anoshift_db_path, f'subset/{year}_subset.parquet'))
    df = df.reset_index(drop=True)
    return df

def rename_columns(df):
    categorical_cols = ["0", "1", "2", "3", "13"]
    numerical_cols = ["4", "5", "6", "7", "8", "9", "10", "11", "12"]
    additional_cols = ["14", "15", "16", "17", "19"]
    label_col = ["18"]

    new_names = []
    for col_name in df.columns.values:
        if col_name in numerical_cols:
            df[col_name] = pd.to_numeric(df[col_name])
            new_names.append((col_name, "num_" + col_name))
        elif col_name in categorical_cols:
            new_names.append((col_name, "cat_" + col_name))
        elif col_name in additional_cols:
            new_names.append((col_name, "bonus_" + col_name))
        elif col_name in label_col:
            df[col_name] = pd.to_numeric(df[col_name])
            new_names.append((col_name, "label"))
        else:
            new_names.append((col_name, col_name))
    df.rename(columns=dict(new_names), inplace=True)
    return df


def preprocess(df, enc=None):
    if not enc:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(df.loc[:, ['cat_' in i for i in df.columns]])

    num_cat_features = enc.transform(df.loc[:, ['cat_' in i for i in df.columns]]).toarray()

    df_catnum = pd.DataFrame(num_cat_features)
    df_catnum = df_catnum.add_prefix('catnum_')

    df = df.reset_index(drop=True)
    df_new = pd.concat([df, df_catnum], axis=1)

    df_new.loc[df_new['label'] < 0, 'label'] = -1
    df_new['label'].replace({1: 0}, inplace=True)
    df_new['label'].replace({-1: 1}, inplace=True)

    return df_new, enc


def get_train(anoshift_db_path, train_data_percent=1):
    dfs = []

    for year in train_years:
        #df_year = load_train_year(anoshift_db_path, year)
        df_year = load(anoshift_db_path, year)
        dfs.append(df_year)

    df_all_years = pd.concat(dfs, ignore_index=True)
    df_all_years = rename_columns(df_all_years)
    df_new, ohe_enc = preprocess(df_all_years)

    num_cols = df_new.columns.to_numpy()[['num_' in i for i in df_new.columns]]
    print(num_cols)

    X_train = df_new[df_new["label"] == 0]
    X_train = X_train.sample(frac=train_data_percent)
    X_train_iso = X_train[num_cols].to_numpy()

    data_mean = X_train_iso.mean(0)[None, :]
    data_std = X_train_iso.std(0)[None, :]
    data_std[data_std == 0] = 1

    return X_train_iso, ohe_enc, data_mean, data_std


def get_n_test_splits():
    return len(test_years)


def get_test(anoshift_db_path, year, ohe_enc, use_valid=False):
    #df_year = load_test_year(anoshift_db_path, year)
    df_year = load(anoshift_db_path, year, use_valid)
    df_year = rename_columns(df_year)
    df_test, _ = preprocess(df_year, ohe_enc)
    isoforest_cols = df_test.columns.to_numpy()[['num_' in i for i in df_test.columns]]
    X_test = df_test[isoforest_cols].to_numpy()
    y_test = df_test["label"].to_numpy()

    X_test = np.nan_to_num(X_test)

    return X_test, y_test

if __name__ == "__main__":
    data_dir = "data/Kyoto-2016_AnoShift"
    X_train_iso, ohe_enc, data_mean, data_std = get_train(data_dir)
    for year in years:
        # original set
        print("original set")
        X_test, y_test = get_test(data_dir, year, ohe_enc)
        X_test, y_test = X_test.astype(np.float32), y_test.astype(int)
        with open(os.path.join(data_dir, "preprocessed", f"{year}.npy"), 'wb') as f:
            np.save(f, X_test)
        with open(os.path.join(data_dir, "preprocessed", f"{year}_label.npy"), 'wb') as f:
            np.save(f, y_test)
        X_test = np.load(os.path.join(data_dir, "preprocessed", f"{year}.npy"))
        y_test = np.load(os.path.join(data_dir, "preprocessed", f"{year}_label.npy"))
        print(f"Year: {year}")
        print(X_test.shape)
        print(y_test.shape)
        del X_test, y_test
        gc.collect()

        # valid set
        print("valid set")
        X_test, y_test = get_test(data_dir, year, ohe_enc, use_valid=True)
        X_test, y_test = X_test.astype(np.float32), y_test.astype(int)
        with open(os.path.join(data_dir, "preprocessed", f"{year}_valid.npy"), 'wb') as f:
            np.save(f, X_test)
        with open(os.path.join(data_dir, "preprocessed", f"{year}_valid_label.npy"), 'wb') as f:
            np.save(f, y_test)
        X_test = np.load(os.path.join(data_dir, "preprocessed", f"{year}_valid.npy"))
        y_test = np.load(os.path.join(data_dir, "preprocessed", f"{year}_valid_label.npy"))
        print(f"Year: {year}")
        print(X_test.shape)
        print(y_test.shape)
        del X_test, y_test
        gc.collect()