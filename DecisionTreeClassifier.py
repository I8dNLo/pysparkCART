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
        self.max_bins = max_bins
        self.fited = False

    def check_df(self, df):
        target_values = df.select(self.target_column).distinct().count()
        if target_values > 2:
            raise ValueError("Too many different value in target")

    def gini(self, df):
        return (df.filter(column > 0.5).count() /df.count()) ** 2 + (df.filter(column <= 0.5).count() /df.count()) ** 2 / df.count()

    def compute_best_split_percentile(self, df, column):

        if df.count() < self.min_samples_leaf:
            return None

        split_field = df.select(column, self.target_column).orderBy(column, ascending=False)
        split_field.cache()

        current_gini = gini(split_field)

        percentile_splits = [i/(self.max_bins) for i in range(1, self.max_bins)]

        splits = map(lambda x: split_field.selectExpr('percentile({}, {})'.format(column, x * 100)), percentile_splits)

        ginies = list(map(lambda x: -current_gini - gini(split_field.filter(column > x)) - gini(split_field.filter(column < x)), splits))

        return splits[np.argmax(ginies)]




    def fit(self, df):
        if self.fited:
            raise PermissionError("Tree already fited")



    def predict_proba(self, df):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

