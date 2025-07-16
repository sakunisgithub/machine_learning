import numpy as np
import pandas as pd
from scipy import stats

class DecisionTreeClassifier :

    def __init__(self, max_depth = None, min_sample = None):
        self.max_depth = max_depth
        self.min_sample = min_sample
        
    def fit(self, X, y) :
        self.tree = self.build_tree(X, y)
    
    def build_tree(self, X, y, depth = 0) :
        node = {
                    "feature_index": None,
                    "threshold": None,
                    "left": None,
                    "right": None,
                    "value": None,
                    "is_leaf": None,
                    "depth": None,
                    "samples": None
                }

        node["samples"] = X.shape[0]
        node["depth"] = depth
        node["value"] = stats.mode(y, keepdims=True).mode[0]

        if len(np.unique(y)) == 1 :
            node["is_leaf"] = 1
            return node
        elif node["samples"] <= self.min_sample :
            node["is_leaf"] = 1
            return node
        elif node["depth"] >= self.max_depth :
            node["is_leaf"] = 1
            return node
        
        best_feature, best_threshold = self.find_best_split(X, y)
    
        if best_feature == None :
            node["is_leaf"] = True
            return node
        
        left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature] > best_threshold)[0]
    
        if len(left_indices) == 0 or len(right_indices) == 0 :
            node["is_leaf"] = True
            return node

        node["is_leaf"] = False
        node["feature_index"] = best_feature
        node["threshold"] = best_threshold

        node["left"] = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        node["right"] = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return node

    def find_best_split(self, X, y) :
        pass