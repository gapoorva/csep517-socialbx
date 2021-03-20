import os

from collections import Counter
from socialbx.sentiment.dataset import Sentiment140TweetDataset
from socialbx.sentiment.word2vec import GenismWord2VecModel
from socialbx.sentiment.encoding import DataEncoder
from socialbx.sentiment.nnmodel import NeuralNetworkSentimentModel

script_dir = os.path.dirname(os.path.realpath(__file__))
pretrained_dir = os.path.join(script_dir, 'pretrained')

dataset = Sentiment140TweetDataset()
training_data = dataset.load_training_data()

word2VecModel = GenismWord2VecModel(
    dataset.train_corpus_dir,
    pretrained_dir
)
word2VecModel.train()

encoder = DataEncoder()
x, y = encoder.encode(training_data['text'], training_data['polarity'])

model = NeuralNetworkSentimentModel(
    word2VecModel,
    encoder,
    pretrained_dir,
)
model.bootstrap()
if not model.trained:
    model.train(x, y)

