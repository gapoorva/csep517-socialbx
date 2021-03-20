from typing import List, Dict, Tuple
import sys
import os
import pprint

import pandas as pd

from socialbx.sentiment.nnmodel import NeuralNetworkSentimentModel
from socialbx.sentiment.encoding import DataEncoder
from socialbx.sentiment.dataset import Sentiment140TweetDataset
from socialbx.sentiment.word2vec import GenismWord2VecModel

Thresholds = Dict[Tuple[float, float], str]

class SentimentPredict(object):
    def __init__(
        self, 
        model: NeuralNetworkSentimentModel,
        encoder: DataEncoder,
        thresholds: Thresholds,
    ):
        self.model = model
        self.encoder = encoder
        self.thresholds = thresholds

    def predict(self, documents: pd.DataFrame):
        x = self.encoder.encodeInputs(documents['text'])

        scores = self.model.predict(x)
        
        results = []
        for i in range(len(documents['text'])):
            results.append({
                'score': scores[i][0],
                'text': documents['original_text'][i],
                'predicted': self.__scoreToHumanLabel(scores[i]),
                'actual': documents['polarity'][i],
            })
        return results


    def __scoreToHumanLabel(self, score):
        for rng, label in self.thresholds.items():
            lower, upper = rng
            if lower < score and score <= upper:
                return label
        return self.thresholds.values()[0]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('usage: predict.py </path/to/csv>')

    file_path = sys.argv[1]
    dataset = Sentiment140TweetDataset()
    inference_data = dataset.load_inference_data(file_path)

    print('found inference data')
    print(inference_data)

    models_dir = os.path.join(os.path.dirname(__file__), 'pretrained')

    word2vecModel = GenismWord2VecModel(None, models_dir)
    encoder = DataEncoder()
    thresholds = {}
    thresholds[(0, 0.6)] = 'NEGATIVE'
    thresholds[(0.6, 1)] = 'POSITIVE'

    sentimentModel = NeuralNetworkSentimentModel(word2vecModel, encoder, models_dir)
    sentimentModel.bootstrap()

    predict = SentimentPredict(sentimentModel, encoder, thresholds)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(predict.predict(inference_data))


