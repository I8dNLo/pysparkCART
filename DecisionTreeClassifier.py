import numpy as np
from pyspark import *
import pyspark as ps
import pandas as pd

conf = pyspark.SparkConf().set('spark.executor.instances',conf_param['num_reducers'])
sc = pyspark.SparkContext.getOrCreate(conf=conf)

class DecissionTreeClassifier:
    def __init__(self, input_columns=None, target_column=None, max_depth=3, min_samples_leaf=1, max_bins=10):
        self.input_columns = input_columns
        self.target_column = target_column
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        max.bins = max_bins

    def check_df(self, df):
        pass

    def fit(self, df):
        pass

    def predict_proba(self, df):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

