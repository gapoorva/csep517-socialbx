import pandas as pd
import os

from nltk.stem import SnowballStemmer
import spacy

from socialbx.sentiment.dataset import Sentiment140TweetDataset
from socialbx.sentiment.word2vec import GenismWord2VecModel
from socialbx.sentiment.encoding import DataEncoder
from socialbx.sentiment.nnmodel import NeuralNetworkSentimentModel
from socialbx.sentiment.predict import SentimentPredict

def recoderFn(x):
    if x == 'negative': 
        return 0
    if x == 'neutral':
        return 0.5
    return 1


raw_data_file_path = os.path.join(os.path.dirname(__file__), 'data', 'Tweets.csv')

print(f"loading data from {raw_data_file_path}")
data = pd.read_csv(raw_data_file_path)

dataset = Sentiment140TweetDataset()
dataset.stemmer = SnowballStemmer('english', ignore_stopwords=True)
data = dataset._preprocess(data, False, 'airline_sentiment', 'text', recoderFn)

data = data.rename(columns={'airline_sentiment': 'polarity'})

models_dir = os.path.join(os.path.dirname(__file__), 'sentiment', 'pretrained')

word2vecModel = GenismWord2VecModel(None, models_dir)
encoder = DataEncoder()
thresholds = {}
thresholds[(0, 0.6)] = 'NEGATIVE'
thresholds[(0.6, 1)] = 'POSITIVE'

sentimentModel = NeuralNetworkSentimentModel(word2vecModel, encoder, models_dir)
sentimentModel.bootstrap()

predict = SentimentPredict(sentimentModel, encoder, thresholds)
predictions = predict.predict(data)

nlp = spacy.load('en_core_web_sm')

summary = {}

for prediction in predictions:
    doc = nlp(prediction['text'])
    for ent in doc.ents:
        if ent.label_ != 'ORG':
            continue
        if ent.text not in summary:
            summary[ent.text] = {'total': 0, 'count': 0}
        summary[ent.text]['total'] += prediction['score']
        summary[ent.text]['count'] += 1

for entity, totals in summary.items():
    print(entity, totals['total'] / totals['count'])