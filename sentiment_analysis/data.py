import csv

from collections import namedtuple


DataPoint = namedtuple('DataPoint', ['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'])


def load_data_points(trainfile):
    print('loading', trainfile, '...')
    data = []
    with open(trainfile) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if 'Sentiment' not in row:
                row['Sentiment'] = None
            dp = DataPoint(**row)
            data.append(dp)
    return data
