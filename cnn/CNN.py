
import numpy as np
import pandas as pd

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Concatenate
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from dataloader.dataloader import DataLoader
from constants.constants import *



class FakeNewsCNN:
    def __init__(self, max_sequence_length= MAX_SEQUENCE_LENGTH, embedding_dim= EMBEDDING_DIM):
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.embeddings_index = None
        self.embeddings_matrix = None
        self.histories = list()

    def get_word_embeddings(self):
        self.embeddings_index = {}
        f = open('../data/glove.6B.100d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            #print(values[1:])
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

    def create_embedding_matrix(self, data: DataLoader):
        self.get_word_embeddings()
        self.embedding_matrix = np.random.random((len(data.word_index) + 1, self.embedding_dim))
        for word, i in data.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    def build_model(self, data: DataLoader):
        self.create_embedding_matrix(data)
        embedding_layer = Embedding(len(data.word_index) + 1,
                                    self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_sequence_length)

        convs = []
        filter_sizes = [3, 4, 5]

        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        for fsz in filter_sizes:
            l_conv = Conv1D(filters=128, kernel_size=fsz, activation='relu')(embedded_sequences)
            l_pool = MaxPooling1D(5)(l_conv)
            convs.append(l_pool)

        l_merge = Concatenate(axis=1)(convs)
        l_cov1 = Conv1D(filters=128, kernel_size=5, activation='relu')(l_merge)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_cov2 = Conv1D(filters=128, kernel_size=5, activation='relu')(l_pool1)
        l_pool2 = MaxPooling1D(30)(l_cov2)
        l_flat = Flatten()(l_pool2)
        l_dense = Dense(128, activation='relu')(l_flat)
        preds = Dense(2, activation='softmax')(l_dense)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['acc'])

    def train(self, data: DataLoader, epochs=3, batch_size=128):

        x_train, x_test, y_train, y_test = train_test_split( data.data, data.labels, test_size=0.20, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split( data.data, data.labels, test_size=0.50, random_state=42)
        if self.model is None:
            raise Exception("Model not built. Call build_model() first.")
        history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                                 epochs=epochs, batch_size=batch_size)
        
        self.histories.append(history)
        return history

    def save_model(self, filename):
        if self.model is None:
            raise Exception("Model not built. Call build_model() first.")
        self.model.save("../models/"+filename)

    def load_model(self, filename):
        self.model = load_model("../models/"+filename)

    def summary(self):
        if self.model is None:
            raise Exception("Model not built. Call build_model() first.")
        self.model.summary()

    def test_model(self, x_test, y_test):
        if self.model is None:
            raise Exception("Model not built. Call build_model() first.")
        
        # Evaluate the model on test data
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print('Test accuracy:', test_acc)
        print('Test loss:', test_loss)
        
        # Predict probabilities for test data
        y_pred_probs = self.model.predict(x_test)
        
        # Convert probabilities to class predictions
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Convert one-hot encoded labels to class labels
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return test_loss, test_acc, cm

    def plot_training_history(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
