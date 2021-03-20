import os

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPool1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

BATCH_SIZE = 2048
EPOCHS = 3

class NeuralNetworkSentimentModel(object):
    def __init__(
        self,
        word2vec_model,
        encoder,
        model_dir,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    ):
        self.word2vec_model = word2vec_model
        self.encoder = encoder
        self.model_path = os.path.join(model_dir, 'sentiment_cnn')
        self.batch_size = batch_size
        self.epochs = epochs
        self.bootstrapped = False
        self.trained = False

    def bootstrap(self):
        print('bootstrapping sentiment model')
        self.bootstrapped = True

        if os.path.exists(self.model_path):
            print('loading pretrained sentiment model from disk')
            self.model = load_model(self.model_path)
            self.trained = True
            return

        print('initializing new sentiment model')

        vocab_size = len(self.encoder.tokenizer.word_index) + 1
        embeddings_size = self.word2vec_model.embeddings_size
        embedding_matrix = np.zeros((vocab_size, embeddings_size))
        for word, i in self.encoder.tokenizer.word_index.items():
            if word in self.word2vec_model.model:
                embedding_matrix[i] = self.word2vec_model.model[word]

        embedding_layer = Embedding(
            vocab_size,
            embeddings_size,
            weights=[embedding_matrix],
            input_length=self.encoder.sequence_length,
            trainable=False,
        )

        self._build_model_architecture(embedding_layer)

    # Edit model architecture here
    def _build_model_architecture(self, embedding_layer):
        self.model = Sequential()
        self.model.add(embedding_layer)
        self.model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        self.model.add(MaxPool1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.summary()

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, x, y):
        print('training sentiment model')

        self.trained = True
        history = self.model.fit(x, y,
            self.batch_size,
            self.epochs,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
                EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)
            ]
        )

        print(history.history)

        self.model.save(self.model_path)
        
    def predict(self, x):
        return self.model.predict(x)

    # TODO: Evaluation



