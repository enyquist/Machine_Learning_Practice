import os

import matplotlib.pyplot as plt
import kaggle
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

#######################################################################################################################
# Global Variables
#######################################################################################################################

HOUSING_PATH = os.path.join('datasets', 'housing')

#######################################################################################################################
# Functions
#######################################################################################################################


def fetch_housing_data(housing_path=HOUSING_PATH):
    """  todo make this work
    Function to fetch housing data. Creates a directory in the cwd and downloads the data
    :param housing_path: the directory to put the data
    :return:
    """
    os.makedirs(housing_path, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('California-Housing-Prices', housing_path, unzip=True)


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Function to load the housing.csv data
    :param housing_path:
    :return:
    """
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def main():
    """
    Main Function
    :return: None
    """
    housing = load_housing_data()

    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                 s=housing['population']/100, label='population', figsize=(10, 7),
                 c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

    plt.legend()

    corr_matrix = housing.corr()

    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

    scatter_matrix(housing[attributes], figsize=(12, 8))


if __name__ == '__main__':
    main()
