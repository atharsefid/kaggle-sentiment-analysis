import csv

from collections import namedtuple


Datapoint = namedtuple("Datapoint", "phraseid sentenceid phrase sentiment")


def load_data_points(trainfile):
    it = csv.reader(open(trainfile, "r"), delimiter="\t")
    row = next(it)  # Drop column names
    if " ".join(row[:3]) != "PhraseId SentenceId Phrase":
        raise ValueError("Input file has wrong column names: {}".format(path))
    data=[]
    for row in it:
        if len(row) == 3:
            row += (None,)
        data.append(Datapoint(*row))
    return data




    '''
    print('loading', trainfile, '...')
    data = []
    with open(trainfile) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if 'Sentiment' not in row:
                row['Sentiment'] = None
            print row
            dp = Datapoint(*row)
            data.append(dp)
    return data
    '''
