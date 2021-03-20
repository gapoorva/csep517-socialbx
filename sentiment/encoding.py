from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from sklearn.manifold import TSNE
# from sklearn.feature_extraction.text import TfidfVectori

SEQUENCE_LENGTH = 300

class DataEncoder(object):
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.tokenizer = Tokenizer()
        self.encoder = LabelEncoder()
        self.sequence_length = sequence_length

    def encodeInputs(self, x_raw):
        print('tokenizing and encoding input data')
        self.tokenizer.fit_on_texts(x_raw.tolist())
        x = pad_sequences(self.tokenizer.texts_to_sequences(x_raw.tolist()), maxlen=SEQUENCE_LENGTH)
        return x

    def encodeLabels(self, y_raw):
        print('encoding labels')
        self.encoder.fit(y_raw.tolist())
        y = self.encoder.transform(y_raw.tolist())

        return y.reshape(-1, 1)

    def encode(self, x_raw, y_raw):
        x = self.encodeInputs(x_raw)
        y = self.encodeLabels(y_raw)
        
        return (x, y)
