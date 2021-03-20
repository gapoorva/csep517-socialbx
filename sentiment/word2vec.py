import os

from gensim import models
from multiprocessing import cpu_count

EMBEDDINGS_SIZE = 300
WORD_WINDOW = 7
MIN_FREQUENCY = 10
EPOCHS = 32

class GenismWord2VecModel(object):
    def __init__(
        self,
        corpus_dir,
        models_dir,
        embeddings_size=EMBEDDINGS_SIZE,
        word_window=WORD_WINDOW,
        min_word_frequency=MIN_FREQUENCY,
        force_train=False,
    ):
        self.corpus_dir = corpus_dir
        self.models_dir = models_dir
        self.embeddings_size = embeddings_size
        self.word_window = word_window
        self.min_word_frequency = min_word_frequency
        self._model = models.Word2Vec()
        self.force_train = force_train
        

    def train(self, epochs=EPOCHS):
        print('training word2vec model')
        pretrained_model_path = os.path.join(self.models_dir, 'word2vec.model')
        if not self.force_train and os.path.exists(pretrained_model_path):
            del self._model
            # load uisng mmap, which makes loading faster, at the tradeoff of no
            # longer being able to train the model further
            print('loading pretrained word2vec model from disk')
            self.model = models.KeyedVectors.load(pretrained_model_path, mmap='r')
            return

        print('training new word2vec model')
        self._model = models.word2vec.Word2Vec(
            sentences=models.word2vec.PathLineSentences(self.corpus_dir),
            size=self.embeddings_size,
            window=self.word_window,
            min_count=self.min_word_frequency,
            workers=cpu_count()*2,
        )

        print('saving word2vec model to disk')        
        self.model = self._model.wv
        self._model.wv.save(fname_or_handle=pretrained_model_path)
        del self._model
