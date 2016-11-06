import math

from sklearn.base import BaseEstimator
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class StatelessTransform(BaseEstimator):

    def fit(self, X, y):
        return self


class ExtractPhrase(StatelessTransform):

    def transform(self, X):
        return [x.Phrase for x in X]


transform_map = {
    'ep': ExtractPhrase,
    'fh': FeatureHasher,
    'cv': CountVectorizer,
    'tfidf': TfidfVectorizer,
    'dtc': DecisionTreeClassifier,
    'rfc': RandomForestClassifier,
    'svc': SVC,
    'gbc': GradientBoostingClassifier,
    'lr': LogisticRegression,
}

param_grids = {
    'fh': [
        {
            'fh__input_type': ['string']
        },
    ],
    'dtc': [
        {
            'dtc__criterion': ['gini', 'entropy'],
            'dtc__min_samples_split': [1,2,3,4],
        },
    ],
    'rfc': [
        {
            'rfc__criterion': ['gini', 'entropy'],
            'rfc__min_samples_split': [1,2,3,4],
        },
    ],
    'svc': [
        {
            'svc__C': [math.pow(10, power) for power in range(-4, 5)],
            'svc__gamma': [math.pow(2, power) for power in range(-4, 5)],
        },
    ],
    'gbc': [
        {
            'gbc__loss': ['deviance', 'exponential'],
            'gbc__min_samples_split': [1,2,3,4],
        },
    ],
    'lr': [
        {
            'lr__solver': ['newton-cg','lbfgs', 'sag'],
            'lr__C': [math.pow(10, power) for power in range(-4, 5)],
        }, {
            'lr__solver': ['liblinear'],
            'lr__penalty': ['l1', 'l2'],
            'lr__C': [math.pow(10, power) for power in range(-4, 5)],
        },
    ],
}


def supported_transforms():
    return list(transform_map.keys())


def get_transform_and_param_grid(name):
    transform = transform_map[name]
    if name in param_grids:
        param_grid = param_grids[name]
    else:
        param_grid = []

    return transform, param_grid
