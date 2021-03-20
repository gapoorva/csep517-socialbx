import os
import re
from multiprocessing import Pool, cpu_count
from itertools import repeat

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

DATA_HEADERS = ['polarity', 'id', 'date', 'query', 'user', 'text']
POLARITY_RECODE = [0, 0, 0.5, 1, 1]
SERIALIZATION_RECODE = {0: 'negative', 0.5: 'neutral', 1: 'positive'}

TWEET_CLEAN_RE = r"@\S+|https?:\S+|[^A-Za-z0-9]+"

def DEFAULT_POLARITY_RECODE(x):
    return POLARITY_RECODE[int(x)]

class Sentiment140TweetDataset(object):
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_dir = os.path.join(current_dir, '../data/sentiment140')
        self.train_corpus_dir = os.path.join(dataset_dir, 'train.processedcorpus')
        self.test_corpus_dir = os.path.join(dataset_dir, 'test.processedcorpus')
        self.train_filepath = os.path.join(dataset_dir, 'train.csv')
        self.test_filepath = os.path.join(dataset_dir, 'test.csv')
        nltk.download('stopwords')

    def load_training_data(self, remove_stopwords=False, force_reprocess=False):
        return self.__load_data(
            self.train_corpus_dir,
            self.train_filepath,
            remove_stopwords,
            force_reprocess,
        )
    
    def load_test_data(self, remove_stopwords=False, force_reprocess=False):
        return self.__load_data(
            self.test_corpus_dir,
            self.test_filepath,
            remove_stopwords,
            force_reprocess,
        )

    def load_inference_data(self, file_path, remove_stopwords=False):
        return self.__load_data(None, file_path, remove_stopwords, True)

    def __load_data(
        self,
        preprocessed_corpus_dir,
        raw_data_file_path,
        remove_stopwords=False,
        force_reprocess=False,
    ):
        '''
        Loads the  data into memory and returns a pandas dataframe
        for the data. The text of tweets are preprocessed, and potentially
        removes patterns or stop words. The labels are recoded from the original
        0-4 score to -1, 0, or 1 where 0-1 is 0 (negative), 2 is 0.5 (neutral),
        and 1 is 1 (positive).
        '''
        print('loading data')
        # check to see if data has already been processed.
        if not force_reprocess:
            data = pd.DataFrame(columns=['polarity','text'])
            data_exists = False
            for label in SERIALIZATION_RECODE.values():
                file_path = os.path.join(preprocessed_corpus_dir, f"{label}.csv")
                if os.path.exists(file_path):
                    data_exists = True
                    df = pd.read_csv(
                        file_path,
                        names=['polarity','text'],
                        dtype={'polarity': np.int32, 'text': np.str}
                    )
                    data = data.append(df)
            
            if data_exists:
                print('loaded preprocessed data from disk:')
                print(data.head(5))
                return data.dropna()

        # load stopwords, if using
        self.stopwords_filter = stopwords.words('english')
        self.stemmer = SnowballStemmer('english', ignore_stopwords=(not remove_stopwords))

        # load data
        print(f"loading data from {raw_data_file_path}")
        data = pd.read_csv(raw_data_file_path, names=DATA_HEADERS)

        print('preprocessing data')        
        n_cores = cpu_count()
        td_split = np.array_split(data, n_cores)
        pool = Pool(n_cores,)
        data = pd.concat(
            pool.starmap(
                self._preprocess, 
                zip(td_split, repeat(remove_stopwords))
            )
        )
        pool.close()
        pool.join()

        print('storing preprocessed data to disk')
        if preprocessed_corpus_dir != None:
            self.store_corpus(data, preprocessed_corpus_dir)

        return data

    def store_corpus(self, df, path):
        if not os.path.exists(path):
            os.makedirs(path)

        grouped = df.groupby('polarity')
        for label in grouped.groups.keys():  
            recoded_label = SERIALIZATION_RECODE[label] 

            grouped.get_group(label).to_csv(
                path_or_buf=os.path.join(path, f"{recoded_label}.csv"),
                columns=['polarity','text'],
                header=False,
                index=False,
            )

    def _preprocess(
        self,
        df,
        remove_stopwords,
        polarity_field='polarity',
        text_field='text',
        recoderFn=DEFAULT_POLARITY_RECODE,
    ):
        def text_preprocess(text):
            # preprocess text
            text = re.sub(TWEET_CLEAN_RE, ' ', str(text).lower().strip())
            tokens = []
            for token in text.split():
                if not remove_stopwords or token not in self.stopwords_filter:
                    stemmed = self.stemmer.stem(token)
                    tokens.append(stemmed)
            processed = ' '.join(tokens)
            if processed.strip() == '':
                #  return a less processed version, to prevent empty tweets
                return str(text).lower().strip()
            return processed
        # recode polarity
        df[polarity_field] = df[polarity_field].apply(recoderFn)
        df['original_text'] = df[text_field]
        df[text_field] = df[text_field].apply(lambda x: text_preprocess(x))
        
        return df

    
