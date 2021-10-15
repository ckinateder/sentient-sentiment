# sentient sentiment v1

import logging
from numpy.random import seed
import re
import os
import sys
import keras
import numpy as np
import pickle
from scipy.interpolate import interp1d
from keras import backend as K
import nltk
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # disable kernel warnings
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

seed(42)
tf.random.set_seed(42)

NUM_CLASSES = 2  # is overwritten based on number of unique vals in data
TEXT_COL = "text"
SCORE_COL = "sentiment"


class V1:
    def __init__(
        self,
        path: str = None,
        epochs: int = 12,
        lstm_units: int = 256,
        embed_out_dim: int = 512,
        num_words: int = 8192,
        batch_size: int = 512,
        test_size: float = 0.2,
    ) -> None:
        # set params
        self.params = {
            "epochs": epochs,
            "lstm_units": lstm_units,
            "embed_out_dim": embed_out_dim,
            "num_words": num_words,
            "batch_size": batch_size,
            "test_size": test_size,
        }  # user settable
        self.MAX_LENGTH = 0  # not user settable

        # get stopwords
        self.get_stop_words()

        # create objects
        self.tokenizer = Tokenizer(
            num_words=self.params["num_words"], split=" "
        )  # OR load from pickle
        self.labelEncoder = LabelEncoder()

        # if loading from path
        if path:
            self.load(path)  # load model and everything if provided path

    def get_stop_words(self):
        try:
            self.stop_words = stopwords.words("english")
        except:
            nltk.download("stopwords")
            nltk.download("wordnet")
            self.stop_words = stopwords.words("english")

    def build_model(self) -> None:
        # model
        assert self.MAX_LENGTH != 0, "params['max_length'] must be set!"
        self.model = Sequential()
        self.model.add(
            Embedding(
                self.params["num_words"],
                self.params["embed_out_dim"],
                input_length=self.MAX_LENGTH,
            )
        )
        self.model.add(SpatialDropout1D(0.4))
        self.model.add(SpatialDropout1D(0.3))
        self.model.add(
            Bidirectional(
                LSTM(
                    self.params["lstm_units"],
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    return_sequences=True,
                )
            )
        )
        self.model.add(
            LSTM(
                int(self.params["lstm_units"] / 2),
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=True,
            )
        )
        self.model.add(
            LSTM(int(self.params["lstm_units"] / 4), dropout=0.2, recurrent_dropout=0.2)
        )
        self.model.add(Dense(NUM_CLASSES, activation="softmax"))
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    def clean(self, values: pd.Series) -> pd.DataFrame:
        """Clean the data for more accurate predictions"""
        values = values.apply(lambda x: str(x).lower())
        # Replace the special characters, digits, numbers
        values = values.apply(lambda x: re.sub("[^A-Za-z]+ ", " ", x))
        # remove html tags
        values = values.apply(lambda x: re.sub("<.*?>", " ", x))
        # lemmatize
        values = values.apply(
            lambda x: " ".join([Word(x).lemmatize() for x in x.split()])
        )
        return values

    def set_max_length(self, sequences: list) -> int:
        """Set the max length when given a list of cleaned sentences"""
        for seq in sequences:
            if len(seq) > self.MAX_LENGTH:
                self.MAX_LENGTH = len(seq)

    def create_X(self, texts: pd.Series):
        """Create the X sequence from a given text series"""
        sequences = self.tokenizer.texts_to_sequences(self.clean(texts).values)
        return pad_sequences(sequences, maxlen=self.MAX_LENGTH)

    def fit(self, data: pd.DataFrame, verbose: int = 1) -> tuple:
        """Fit the model on the given data"""
        assert (
            TEXT_COL in data.columns and SCORE_COL in data.columns
        ), "TEXT_COL/SCORE_COL not in data!"
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
            X,
            y,
            test_size=self.params["test_size"],
            random_state=42,
        )

        # build model to length size
        self.build_model()

        # fit model with params
        logging.info(
            f"Fitting model with params {self.params} on dataset with shape {X_train.shape}"
        )
        self.model.fit(
            X_train,
            y_train,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            verbose=verbose,
        )

        acc_and_loss = self.model.evaluate(X_test, y_test)
        loss = acc_and_loss[0]
        accuracy = acc_and_loss[1]
        logging.info(f"Model loss on test set with {y_test.shape[0]} rows: {loss:.4f}")
        logging.info(
            f"Model accuracy on test set with {y_test.shape[0]} rows: {accuracy*100:0.2f}%"
        )
        return loss, accuracy

    def save(self, path: str) -> None:
        """Takes a directory and saves both the model and tokenizer
        Tree:
        path
        --model
        --tokenizer
        """
        try:
            if not os.path.exists(path):
                os.mkdir(path)
            with open(os.path.join(path, "tokenizer"), "w+b") as f:
                pickle.dump(self.tokenizer, f)
            with open(os.path.join(path, "max_len"), "w+b") as f:
                pickle.dump(self.MAX_LENGTH, f)
            self.model.save(os.path.join(path, "model"), overwrite=True)
            logging.info(f"Saved model to '{path}'")
        except:
            logging.warning(f"Could not save model to '{path}'")

    def load(self, path: str) -> None:
        """Takes a directory and loads both the model and tokenizer"""
        try:
            with open(os.path.join(path, "tokenizer"), "rb") as f:
                self.tokenizer = pickle.load(f)
            with open(os.path.join(path, "max_len"), "rb") as f:
                self.MAX_LENGTH = pickle.load(f)
            self.model = keras.models.load_model(os.path.join(path, "model"))
            logging.info(f"Loaded model from '{path}'")
        except:
            logging.error(f"Could not load model from '{path}'")
            sys.exit(1)

    def score(
        self, *args
    ) -> dict:  # predict score for a text input, and return % positive
        interp = interp1d([0, 1], [-1, 1])
        values = {}
        y = self.model.predict(self.create_X(pd.Series([*args])))
        for i in range(len(args)):
            values[args[i]] = round(float(interp(y[i][1])), 4)
        return values
