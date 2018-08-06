import pandas as import pd
import numpy as np
import os


class provSeries(pandas.Series):
    @property
    def _constructor(self):
        return provSeries

    def custom_series_function(self):
        return 'OK'

class ProvDataFrame(pandas.DataFrame):
    """
    a subclass extending pandas DataFrames to contain meta data about the
    generating process
    """

    def __init__(self, *args, **kw):
        super(ProvDataFrame, self).__init__(*args, **kw)

    @property
    def _constructor(self):
        return ProvDataFrame

    _constructor_sliced = provSeries

    def to_csv(self, filename, path=os.pwd):
        """
        save to two files, in a folder one _info.txt and one _data.csv

        filename should not include any extension
        """
        full_path = os.path.join(path,filename)
        os.mkdir(path,filename)

        self.to_csv(os.path.join(full_path,filename,'_data.csv'))

    def load(self,filename, path=os.pwd):
        """
        load from data saved previously as this type of object
        """
        full_path = os.path.join(path,filename)

        self.df = pd.read_csv()
