import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from constants.constants import *


import os, re

class DataLoader:

    def __init__(self, filename) -> None:

        self.raw_df = pd.read_csv("../data/" + filename)
        sequences, labels, self.word_index = self.get_metadata()
        self.data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        self.labels = to_categorical(np.asarray(labels),num_classes = 2)


    def create_df(self, name):
        pass

    def clean_str(self, string):

        string = re.sub(r"\n", " ", string)
        string = re.sub(r"\\", "", string)    
        string = re.sub(r"\'", "", string)    
        string = re.sub(r"\"", "", string)    
        string = re.sub(r"\*{2}.*?\*{2}", "", string)  # Remove text between ** **
        string = re.sub(r"\*.*?\*", "", string)        # Remove text between * *
        return string.strip().lower()

    def get_metadata(self):

        texts = []
        labels = []

        pattern1 = r"\*{2}.*?\*{2}"  # Remove text between ** **
        pattern2 = r"\*.*?\*"         # Remove text between * *

        # Apply the regular expressions to the 'text' column
        self.raw_df['text'] = self.raw_df['text'].str.replace(pattern1, "", regex=True)
        self.raw_df['text'] = self.raw_df['text'].str.replace(pattern2, "", regex=True)
        texts = self.raw_df['text'].astype(str)
        if 'title' in self.raw_df.columns:
            texts = self.raw_df['title'].astype(str) + ' ' + texts
        labels = self.raw_df['label']


            
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        return sequences, labels, word_index
