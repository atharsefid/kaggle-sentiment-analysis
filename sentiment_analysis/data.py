import csv


def load_training_data(trainfile):
    print('loading', trainfile, '...')
    X = []
    y = []
    with open(trainfile) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            X.append(row['Phrase'])
            y.append(row['Sentiment'])
    return X, y
