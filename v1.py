# sentient sentiment v1

import pickle
from scipy.interpolate import interp1d
from keras import backend as K
import nltk
from operator import xor
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
from pathlib import Path
import os
import re
import jsonlines
from pprint import pprint
from numpy.random import seed
seed(42)
tf.random.set_seed(42)
nltk.download("stopwords")
nltk.download("wordnet")

NUM_WORDS = 8192  # number of unique words #(10000)
LSTM_UNITS = 256  # lstm cells #(300)
EMBED_OUT_DIM = 512
NUM_CLASSES = 2  # is overwritten based on number of unique vals in data
NUM_EPOCHS = 8  # keep lower to avoid overfitting
TEXT_COL = "text"
SCORE_COL = "sentiment"
BATCH_SIZE = 512  # (400)
TEST_SIZE = 0.2  #

# get stop words
stop_words = stopwords.words("english")


class V1:
    def __init__(self, path=None) -> None:
        # tokenizer and encoderpip3 inst
        self.tokenizer = Tokenizer(
            num_words=NUM_WORDS, split=' ')  # OR load from pickle
        self.labelEncoder = LabelEncoder()
        self.MAX_LENGTH = 0
        if path:
            pass

    def build_model(self) -> None:
        # model
        self.model = Sequential()
        self.model.add(
            Embedding(NUM_WORDS, EMBED_OUT_DIM, input_length=self.MAX_LENGTH))
        self.model.add(SpatialDropout1D(0.4))
        self.model.add(SpatialDropout1D(0.3))
        self.model.add(Bidirectional(
            LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        self.model.add(LSTM(int(LSTM_UNITS/2), dropout=0.2,
                       recurrent_dropout=0.2, return_sequences=True))
        self.model.add(
            LSTM(int(LSTM_UNITS/4), dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(NUM_CLASSES, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam", metrics=["accuracy"])

    def clean(self, values: pd.Series) -> pd.DataFrame:
        # make all lowercase
        # ' '.join(x.lower() for x in x.split()))
        values = values.apply(lambda x: str(x).lower())
        # Replacing the special characters, digits, numbers
        values = values.apply(lambda x: re.sub('[^A-Za-z]+ ', ' ', x))
        # remove html tags
        values = values.apply(lambda x: re.sub('<.*?>', ' ', x))
        # Lemmatization
        values = values.apply(lambda x: ' '.join(
            [Word(x).lemmatize() for x in x.split()]))
        return values

    def set_max_length(self, sequences: list) -> int:
        for seq in sequences:
            if len(seq) > self.MAX_LENGTH:
                self.MAX_LENGTH = len(seq)

    def create_X(self, texts: pd.Series):
        sequences = self.tokenizer.texts_to_sequences(self.clean(texts).values)
        return pad_sequences(sequences, maxlen=self.MAX_LENGTH)

    def fit(self, data, verbose: int = 1) -> tuple:
        # data[SCORE_COL] = self.labelEncoder.fit_transform(data[SCORE_COL])
        data[TEXT_COL] = self.clean(data[TEXT_COL])
        self.tokenizer.fit_on_texts(data[TEXT_COL].values)

        # create sequences
        sequences = self.tokenizer.texts_to_sequences(data[TEXT_COL].values)

        # find max length
        self.set_max_length(sequences)

        # set max length here
        X = pad_sequences(sequences, maxlen=self.MAX_LENGTH)
        y = pd.get_dummies(data[SCORE_COL])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, )

        # build model to length size
        self.build_model()

        self.model.fit(X_train, y_train, epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE, verbose=verbose)

        acc_and_loss = self.model.evaluate(X_test, y_test)
        loss = acc_and_loss[0]
        accuracy = acc_and_loss[1]
        print(
            f"Model loss on test set with {y_test.shape[0]} rows: {loss:.4f}")
        print(
            f"Model accuracy on test set with {y_test.shape[0]} rows: {accuracy*100:0.2f}%")
        return loss, accuracy

    def save(self, path: str) -> None:
        """Takes a directory and saves both the model and tokenizer
        Tree:
        path
        --model
        --tokenizer
        """
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, "tokenizer"), "w+b") as f:
            pickle.dump(self.tokenizer, f)
        with open(os.path.join(path, "max_len"), "w+b") as f:
            pickle.dump(self.MAX_LENGTH, f)
        self.model.save(os.path.join(path, "model"), overwrite=True)
        print("Saved model to", path)

    def load(self, path: str) -> None:
        """Takes a directory and loads both the model and tokenizer
        """
        with open(os.path.join(path, "tokenizer"), "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(os.path.join(path, "max_len"), "rb") as f:
            self.MAX_LENGTH = pickle.load(f)
        self.model = keras.models.load_model(os.path.join(path, "model"))
        print("Loaded model from", path)

    def score(self, *args) -> dict:  # predict score for a text input, and return % positive
        interp = interp1d([0, 1], [-1, 1])
        values = {}
        y = self.model.predict(self.create_X(pd.Series([*args])))
        for i in range(len(args)):
            values[args[i]] = round(float(interp(y[i][1])), 4)
        return values


def load_data():
    """load datasets"""
    def read_jsonl(path: str) -> pd.DataFrame:
        out = []
        with jsonlines.open(path) as f:
            for line in f.iter():
                out.append(line)
        return pd.DataFrame(out)

    def filter(df: pd.DataFrame, col: str):
        df[col] = np.select(
            [df[col] > 3, df[col] == 3, df[col] < 3], [
                "positive", "neutral", "negative"]
        )
    labelEncoder = LabelEncoder()

    # from http://jmcauley.ucsd.edu/data/amazon/
    amazon_set_1 = read_jsonl("datasets/reviews_Musical_Instruments_5.json")[
        ["summary", "overall"]].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_1, SCORE_COL)

    amazon_set_2 = read_jsonl("datasets/reviews_Office_Products_5.json")[
        ["summary", "overall"]].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_2, SCORE_COL)

    amazon_set_3 = read_jsonl("datasets/reviews_Tools_and_Home_Improvement_5.json")[
        ["summary", "overall"]].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_3, SCORE_COL)

    amazon_set_4 = read_jsonl("datasets/reviews_Toys_and_Games_5.json")[
        ["summary", "overall"]].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_4, SCORE_COL)

    amazon_set_5 = read_jsonl("datasets/reviews_Home_and_Kitchen_5.json")[
        ["summary", "overall"]].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_5, SCORE_COL)

    finance_set_1 = pd.read_csv(
        "datasets/small/all-data.csv", encoding="ISO-8859-1")
    finance_set_1.columns = [SCORE_COL, TEXT_COL]

    data = pd.concat([finance_set_1, amazon_set_1, amazon_set_2,
                     amazon_set_3, amazon_set_4, amazon_set_5])
    # remove neutral rows cause they do not matter
    data = data[data[SCORE_COL] != "neutral"]
    data = data.sample(frac=1).reset_index(drop=True)  # shuffle
    data[SCORE_COL] = labelEncoder.fit_transform(data[SCORE_COL])
    return data


def equalize_distribution(data: pd.DataFrame) -> pd.DataFrame:
    pos_count = data[data[SCORE_COL] == 'positive'].shape[0]
    neg_count = data[data[SCORE_COL] == 'negative'].shape[0]
    if pos_count > neg_count:
        data[data[SCORE_COL] == 'positive'] = data[data[SCORE_COL]
                                                   == 'positive'][-neg_count:]
        data = data.dropna()
    elif neg_count > pos_count:
        data[data[SCORE_COL] == 'negative'] = data[data[SCORE_COL]
                                                   == 'negative'][-pos_count:]
        data = data.dropna()
    return data


if __name__ == "__main__":
    NUM_EPOCHS = 10
    v1 = V1()
    #data = equalize_distribution(load_data())

    v1.load("trained107")
    pprint(v1.score("Apple's sales dropped 10 percent today", "the most amazing product",
                    "your mom is the best", "i love your mom", "you're the worst"))


"""
To put in writeup:
inputs must  be in format
outputs will be  same way
where saving happens
sample full cycle
"""
