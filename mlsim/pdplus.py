import pandas as pd
import numpy as np
import os

# following instructions here:
# https://pandas.pydata.org/pandas-docs/stable/development/extending.html#subclassing-pandas-data-structures
def load(self,filename, path=os.pwd):
    """
    load from data saved previously as this type of object
    """
    full_path = os.path.join(path,filename)

    self = pd.read_csv()


class provSeries(pd.Series):
    """
    (TO COMPLETE) make this extend a series to be strucutred and contain all of
    the information about the data generation
    """

    @property
    def _constructor(self):
        return provSeries

    @property
    def _constructor_expanddim(self):
        return ProvDataFrame

    # def custom_series_function(self):
    #     return 'OK'

    _metadata = ['provenance']

class ProvDataFrame(pd.DataFrame):
    """
    a subclass extending pandas DataFrames to contain meta data about the
    generating process

    (TO COMPLETE) make this have an extra feild, the provSeries, that contains
    the information about how the data was generated
    """

    def __init__(self, *args, **kw):
        super(ProvDataFrame, self).__init__(*args, **kw)

    @property
    def _constructor(self):
        return ProvDataFrame

    @property
    def _constructor_sliced(self):
        return ProvSeries

    _metadata = ['provenance']



    def to_csv(self, filename, path=os.pwd):
        """
        save to two files, in a folder one _info.txt and one _data.csv

        filename should not include any extension
        """
        full_path = os.path.join(path,filename)
        os.mkdir(path,filename)

        self.to_csv(os.path.join(full_path,filename,'_data.csv'))
        self.provenance.to_csv(os.path.join(full_path,filename,'_provenance.csv'))
