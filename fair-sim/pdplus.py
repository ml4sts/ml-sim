import pandas as import pd
import numpy as np


class pdplus:
    """
    a wrapper of pandas DataFrames that includes extra info for reproducibitliy
    and writes out that information as well

    accessing pdplus.df directly gives a pandas dataframe
    """

    def __init__(self,data,columns):
        self.df = pd.DataFrame(data,columns)
        self.seed = np

    def save_pair(self, filename):
        """
        save to two files, one _info.txt and one _data.csv

        filename should be full path, with no extension
        """
        self.df.to_csv(filename+'_data.csv')

    def save(self, filename):
        """
        save to a txt file that's~ two csv files

        filename should be full path, with no extension
        """

        self.df.to_csv('tmp.csv')

        with open('tmp.csv', 'r') as dat_only:
            data = dat_only.read()

        # for prepending other data as
        with open(filename, 'w') as full:
            full.write("new first line\n" + data)
