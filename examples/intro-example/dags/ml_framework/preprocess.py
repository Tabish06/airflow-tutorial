import numpy as np
import pandas as pd
# import sys,os
# sys.path.append('../')

from sklearn.model_selection import StratifiedKFold

# import shap
# this is z-score that value minus mean divided by standard deviation
# http://duramecho.com/Misc/WhyMinusOneInSd.html
def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def min_max_normalize(dataset):
    maxVal = np.max(dataset,axis=0)
    minVal = np.min(dataset,axis=0)
    return ( (dataset - minVal) / (maxVal - minVal))

def str_to_int(df):
    str_columns = df.select_dtypes(['object']).columns
    print(str_columns)
    for col in str_columns:
        df[col] = df[col].astype('category')

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df


def count_space_except_nan(x):
    if isinstance(x,str):
        return x.count(" ") + 1
    else :
        return 0

# https://stackoverflow.com/a/42523230
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:

        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        del df[each]
        df = pd.concat([df, dummies], axis=1)
    return df



def preprocess(kfold_cross_validation_count,file_path,y_pred_column,y_pred_values,delimiter_specified,one_hot_columns,drop_columns,min_max):
    # try :
    # print(os.getcwd())
    data = pd.read_csv(file_path,delimiter=delimiter_specified)
    data = one_hot(data,data.loc[:,one_hot_columns].columns)
    data = str_to_int(data)
    dataY = data.loc[:,[f'{y_pred_column}_{y_pred_values[0]}',f'{y_pred_column}_{y_pred_values[1]}']]
    data = data.drop(f'{y_pred_column}_{y_pred_values[0]}',axis=1)
    data = data.drop(f'{y_pred_column}_{y_pred_values[1]}',axis=1)

    for column in drop_columns :
        if column != '':
           data.drop(column,axis=1)
    # print(data)
    if min_max :
        data = min_max_normalize(data)
    else :
        data = feature_normalize(data)
    # data.to_csv('data.csv', sep='\t')
    ratio = ((dataY.iloc[:,[1]] == 1 ).sum()/ len(dataY))
    kfold_fit(data,dataY.iloc[:,[1]],kfold_cross_validation_count)
    print(f'Ratio : {ratio}')
    return ratio
    # except :
    #     logging.error('Failed to import csv ', os.getcwd())
def kfold_fit(train,trainY,splits):
    kf = StratifiedKFold(n_splits=splits,shuffle=True, random_state = 13)
    kf.get_n_splits(train,trainY)
    i = 0
    for (train_idx,test_idx) in kf.split(train,trainY):
        train.iloc[train_idx].to_csv(f'train_{i}.csv')
        trainY.iloc[train_idx].to_csv(f'trainY_{i}.csv')
        train.iloc[test_idx].to_csv(f'test_{i}.csv')
        trainY.iloc[test_idx].to_csv(f'testY_{i}.csv')
        i = i + 1