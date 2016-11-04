#!/usr/bin/env python3
"""
Find the best estimator
"""

import argparse
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import sentiment_analysis.data as data


def main(args):
    X, y = data.load_training_data(args.train)
    #print(X)
    #print(y)
    print('setup pipeline...')
    steps = [('cv', CountVectorizer()), ('svc', SVC())]
    pipeline = Pipeline(steps)

    print('grid search...')
    param_grid = {
        'svc__C': [math.pow(10, power) for power in range(-4, 5)]
    }

    gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, verbose=args.verbose).fit(X, y)
    print("best params = %s\nbest score = %f\n" % (gs.best_params_, gs.best_score_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='train.tsv', help='Training data in TSV format')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='Verbosity')

    args = parser.parse_args()
    main(args)
