#!/usr/bin/env python3
"""
Find the best estimator
"""

import argparse

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sentiment_analysis import data
from sentiment_analysis import transforms


def main(args):
    X, y = data.load_training_data(args.train)

    print('setup pipeline...')
    steps = []
    param_grid = []
    for step in args.steps:
        transform, pg = transforms.get_transform_and_param_grid(step)
        steps.append((step, transform()))
        param_grid.extend(pg)

    pipeline = Pipeline(steps)

    print('grid search...')
    gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, verbose=args.verbose).fit(X, y)
    print('best params =', gs.best_params_)
    print('best score =', gs.best_score_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='train.tsv', help='Training data in TSV format')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='Verbosity')
    parser.add_argument('-s', '--steps', required=True, nargs='*', help="Steps in the pipeline, e.g. cv svc. Valid steps are: %s" % ', '.join(transforms.supported_transforms()))

    args = parser.parse_args()
    main(args)
