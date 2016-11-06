#!/usr/bin/env python3
"""
Find the best parameters for estimator pipeline and classify the sentiment of phrases
"""

import argparse
import csv

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sentiment_analysis import data
from sentiment_analysis import transforms


def main(args):
    X_train = data.load_data_points(args.train)
    y_train = [x.Sentiment for x in X_train]

    print("setup pipeline: %s..." % args.steps)
    steps = []
    param_grid = []
    for step in args.steps:
        transform, pg = transforms.get_transform_and_param_grid(step)
        steps.append((step, transform()))

        new_param_grid = []
        if not param_grid:
            new_param_grid = pg
        else:
            for outer_dict in param_grid:
                for inner_dict in pg:
                    merged_dict = {**outer_dict, **inner_dict}
                    new_param_grid.append(merged_dict)
        param_grid = new_param_grid

    pipeline = Pipeline(steps)

    print('grid search...')
    gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, verbose=args.verbose).fit(X_train, y_train)
    print('best params =', gs.best_params_)
    print('best score =', gs.best_score_)

    # predict on test data
    X_test = data.load_data_points(args.test)
    print('make predictions...')
    y_test = gs.predict(X_test)

    # output result to CSV file
    print("outout result to %s..." % args.outfile)
    phrase_ids = [x.PhraseId for x in X_test]
    results = zip(phrase_ids, y_test)
    with open(args.outfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['PhraseId', 'Sentiment'])
        for r in results:
            writer.writerow(r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-T', '--train', default='train.tsv', help='Training data in TSV format')
    parser.add_argument('-s', '--steps', required=True, nargs='*', help="Steps in the pipeline, e.g. ep cv dtc. Valid steps are: %s" % ', '.join(transforms.supported_transforms()))
    parser.add_argument('-t', '--test', default='test.tsv', help='Test data in TSV format')
    parser.add_argument('-o', '--outfile', required=True, help='Output file in CSV format')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='Verbosity')

    args = parser.parse_args()
    main(args)
