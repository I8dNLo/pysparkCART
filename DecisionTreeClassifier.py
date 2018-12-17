import numpy as np
from pyspark import *
import pyspark as ps
import pandas as pd
from pyspark.sql.functions import udf
import json
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
        self.tree = {"weight" : None}

    def check_df(self, df):
        target_values = df.select(self.target_column).distinct().count()
        if target_values > 2:
            raise ValueError("Too many different value in target")

    def gini(self, df):
        return (df.filter(column > 0.5).count() /df.count()) ** 2 + (df.filter(column <= 0.5).count() /df.count()) ** 2 / df.count()

    def compute_best_split_percentile_column(self, df, column):

        if df.count() < self.min_samples_leaf:
            return None

        split_field = df.select(column, self.target_column).orderBy(column, ascending=False)
        split_field.cache()

        current_gini = gini(split_field)

        percentile_splits = [i/(self.max_bins) for i in range(1, self.max_bins)]

        splits = map(lambda x: split_field.selectExpr('percentile({}, {})'.format(column, x * 100)), percentile_splits)

        gini_scores = list(map(lambda x: -current_gini - gini(split_field.filter(column > x)) - gini(split_field.filter(column < x)), splits))

        return splits[np.argmax(gini_scores)]

    def best_split(self, df):
        splits = list(map(lambda x: self.compute_best_split_percentile_column(df, x),  self.input_columns))

        best_split_column = self.input_columns(np.agrmax(splits))
        return (best_split_column, self.compute_best_split_percentile_column(df, best_split_column))

    def build_node(self, df, depth):
        mean_leaf = df.agg({self.target_column: "avg"})
        if df.count() <= self.min_samples_leaf or depth >= self.max_depth or mean_leaf == 1.0 or mean_leaf == 0.0:
            return {"wight" : df.agg({self.target_column: "avg"})}
        else:
            col, split = best_split(df)

            node = {"attr" : col,
                    "cutoff" : split,
                    "left" : self.build_node(df.filter(col < split, depth+1))
                    "right" : self.build_node((df.filter(col >= split, depth+1)))}
            return node

    def build_tree(self, df):
        return self.build_node(df, 0)

    def fit(self, df):
        if self.fited:
            raise PermissionError("Tree already fited")

        self.tree = self.build_tree(df)
        self.fited = True


    def object_score(self, obj, node):
        if node.get("weight", False):
            return node['weight']
        elif obj[node['attr']]<node['cutoff']:
            return self.object_score(self, obj, node['left'])
        else:
            return self.object_score(self, obj, node['right'])
    def tree_predict(self, obj):
        return self.object_score(obj, self.tree)

    def predict_proba(self, df):
        if self.fited:
            tree_udf = udf(lambda: object_score)
            results = df.withColumn("probability", tree_udf(*self.input_columns))
            return results.select('probability')
        else:
            raise ValueError("Tree not fitted!")

    def save(self, filename):
        if self.fited:
            with open(filename, "w") as f:
                f.write(json.dumps(self.tree))
        else:
            raise ValueError("Tree not fited!")

    def load(self, filename):
        if not self.fited:
            with open(filename, "r") as f:
                self.tree = json.load(f)
        else:
            raise ValueError("Tree is already fited!")
