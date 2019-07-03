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



def preprocess(kfold_cross_validation_count,file_path,y_pred_column,y_pred_values,delimiter_specified,one_hot_columns,drop_columns):
    # try :
    # print(os.getcwd())
    bank_marketing = pd.read_csv(file_path,delimiter=delimiter_specified)
    bank_marketing = one_hot(bank_marketing,bank_marketing.loc[:,one_hot_columns].columns)
    bank_marketing = str_to_int(bank_marketing)
    bank_marketingY = bank_marketing.loc[:,[f'{y_pred_column}_{y_pred_values[0]}',f'{y_pred_column}_{y_pred_values[1]}']]
    bank_marketing = bank_marketing.drop(f'{y_pred_column}_{y_pred_values[0]}',axis=1)
    bank_marketing = bank_marketing.drop(f'{y_pred_column}_{y_pred_values[1]}',axis=1)

    for column in drop_columns :
        bank_marketing.drop(column,axis=1)
    # print(bank_marketing)

    bank_marketing = feature_normalize(bank_marketing)
    # bank_marketing.to_csv('bank_marketing.csv', sep='\t')
    ratio = ((bank_marketingY.iloc[:,[1]] == 1 ).sum()/ len(bank_marketingY))
    kfold_fit(bank_marketing,bank_marketingY.iloc[:,[1]],kfold_cross_validation_count)
    print(f'Ratio : {ratio}')
    return ratio
    # except :
    #     logging.error('Failed to import csv ', os.getcwd())
def kfold_fit(train,trainY,splits):
    kf = StratifiedKFold(n_splits=splits,shuffle=True, random_state = 5)
    kf.get_n_splits(train,trainY)
    i = 0
    for (train_idx,test_idx) in kf.split(train,trainY):
        train.iloc[train_idx].to_csv(f'train_{i}.csv')
        trainY.iloc[train_idx].to_csv(f'trainY_{i}.csv')
        train.iloc[test_idx].to_csv(f'test_{i}.csv')
        trainY.iloc[test_idx].to_csv(f'testY_{i}.csv')
        i = i + 1