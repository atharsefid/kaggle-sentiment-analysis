#!/usr/bin/env python3
"""
Classify the sentiment of sentences from the Rotten Tomatoes dataset

https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
"""

import argparse
import csv
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def load_training_data(trainfile):
    X = []
    y = []
    with open(trainfile) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            X.append(row['Phrase'])
            y.append(row['Sentiment'])
    return X, y


def main(args):
    X_train, y_train = load_training_data(args.train)
    #print(X_train)
    #print(y_train)
    pipeline = make_pipeline(CountVectorizer(), SVC())
    pipeline.fit(X_train, y_train)
    # 5-fold
    scores = cross_validation.cross_val_score(pipeline, X_train, y_train, cv=5)
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='train.tsv', help='Training data in TSV format')
    parser.add_argument('--test', default='test.tsv', help='Test data in TSV format')

    args = parser.parse_args()
    main(args)
