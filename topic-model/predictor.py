from __future__ import print_function
import os
import json
import pickle
import io as StringIO
import sys
import signal
import traceback
import flask
import numpy as np
import pandas as pd

class ScoringService(object):
    '''
    Takes a tidy (ideally stemmed) numpy array of text and
    returns topic probabilities based off of keyword
    heuristic in a data-frame format.
    '''
    
    _TOPIC_NAMES = ['Reservoir', 'Trap', 'Charge', 'Seal', 'Other']

    _TOPIC_TERMS = [['porosity', 'permeability', 'sorting', 'maturity', 'reservoir', 'diagenesis', 'facies', 'lithofacies'],
                       ['trap', 'closure', 'structure', 'timing', 'traps'],
                       ['maturity', 'migration', 'charge', 'dhi', 'expulsion', 'show', 'seeps', 'toc', 'kerogen'],
                       ['seal', 'caprock', 'pressure', 'faulted', 'thief', 'leaky']
                      ]

    @classmethod
    def _search_terms(cls, df_row, topic_idx):
        count = 0
        for term in cls._TOPIC_TERMS[topic_idx]:
            if term in df_row.Text:
                count += 1
        return(count)

    @classmethod
    def _search_other(cls, df_row):
        if df_row.loc[cls._TOPIC_NAMES[:-1]].sum() == 0:
            return(1)
        else:
            return(0)

    @classmethod
    def _normalize(cls, df_row):
        return(df_row[cls._TOPIC_NAMES] / df_row[cls._TOPIC_NAMES].sum())

    @classmethod
    def predict(cls, input):
        docs = pd.Series(input, index=range(len(input))).to_frame('Text')
        doc_topic_dist = pd.DataFrame()
        for idx, topic in enumerate(cls._TOPIC_NAMES[:-1]):
            doc_topic_dist[topic] = docs.apply(cls._search_terms, topic_idx = idx, axis=1)
        doc_topic_dist['Other'] = doc_topic_dist.apply(cls._search_other, axis=1)
        doc_topic_dist = doc_topic_dist.apply(cls._normalize, axis=1)
        doc_topic_dist['Topic'] = doc_topic_dist.apply(np.argmax, axis=1)
        return doc_topic_dist

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = True

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data.values[:,0])

    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    predictions.to_csv(out, header=True, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
