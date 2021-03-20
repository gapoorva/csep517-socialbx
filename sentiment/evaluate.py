import sys
import os
import pprint

import pandas as pd

from socialbx.sentiment.nnmodel import NeuralNetworkSentimentModel
from socialbx.sentiment.encoding import DataEncoder
from socialbx.sentiment.dataset import Sentiment140TweetDataset
from socialbx.sentiment.word2vec import GenismWord2VecModel

file_path = os.path.join(os.path.dirname(__file__), '../data/sentiment140/test.csv')
dataset = Sentiment140TweetDataset()
test_data = dataset.load_inference_data(file_path)

print('found test data')
print(test_data)

models_dir = os.path.join(os.path.dirname(__file__), 'pretrained')

word2vecModel = GenismWord2VecModel(None, models_dir)
encoder = DataEncoder()
thresholds = {}
thresholds[(0, 0.6)] = 'NEGATIVE'
thresholds[(0.6, 1)] = 'POSITIVE'

sentimentModel = NeuralNetworkSentimentModel(word2vecModel, encoder, models_dir)
sentimentModel.bootstrap()

x, y = encoder.encode(test_data['text'], test_data['polarity'])

results = sentimentModel.model.evaluate(x, y, batch_size=2048)

print('loss', results[0])
print('accuracy', results[1])
