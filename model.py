# -*- coding: utf-8 -*-
from __future__ import unicode_literals

'''
Created on April 22, 2020
@author: Anais Ollagnier

'''

import random, joblib, pickle
import numpy as np
from collections import Counter

import tensorflow as tf
import tensorflow_hub as hub
print("Version: ", tf.__version__)
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from cnn_model import CNN
import utils

class OneVSRest:
    def __init__(
            self, dirModel, options
            ):
        utils.reset_seeds()
        self.dirModel=dirModel
        self.options= options

        self.args = {
                # WORD-level
                'MAX_NUM_WORDS':10000,
                'EMBEDDING_DIM':300,
                'MAX_LEN': 396,

                # LEARNING
                'BATCH_SIZE':32,
                'DROP_OUT' : 0.2,
                'ATTENTION': False,
                'NB_EPOCHS':8,
                'RUNS':5, ##
                'VAL_SIZE':0.2
                }
    
    def _runTrain(self, dataFrame):
        
        model = Model(self.dirModel, self.options)
        model._tokenizer_initialization(dataFrame, word_vocab_size=self.args['MAX_NUM_WORDS'])
        nb_classes=1
        embedding_layer=None

        for column in dataFrame.columns[1:]:
            df=dataFrame.filter(items=['text', column])
            train_x, train_y=model._preprocess(df, model.w2idx_dict, self.args['MAX_LEN']), df.iloc[:,-1]
            print('Build {} model, classes distribution: {} ...'.format(column, Counter(train_y)))

            model=CNN(
                    embedding_layer=embedding_layer,
                    num_words=self.args['MAX_NUM_WORDS'],
                    embedding_dim=self.args['EMBEDDING_DIM'],
                    max_seq_length=self.args['MAX_LEN'],
                    kernel_sizes=[3],
                    feature_maps=[100],
                    nb_classes=nb_classes,
                    activation='sigmoid'
                    ).build_model()


            model.compile(
                    loss='binary_crossentropy',
                    optimizer=tf.optimizers.Adam(),
                    metrics=['accuracy']
                    )

            model.summary()

            model.fit(
                    train_x, train_y,
                    epochs=self.args['NB_EPOCHS'],
                    batch_size=self.args['BATCH_SIZE']
                    )

            ## save model
            model.save(self.dirModel+'/'+column+'.h5')
            print('Model saved into: ', self.dirModel+'/'+column+'.h5')
            del model, df
            gc.collect()


class Model:
    def __init__(
            self, dirModel, options
            ):
        
        utils.reset_seeds()
        self.dirModel=dirModel

        self.w2idx_dict = None

        self.args = {
                # WORD-level
                'MAX_NUM_WORDS':10000,
                'EMBEDDING_DIM':300,
                'MAX_LEN': 396,

                # LEARNING
                'BATCH_SIZE':32,
                'DROP_OUT' : 0.2,
                'ATTENTION': False,
                'NB_EPOCHS':8,
                'RUNS':5, ##
                'VAL_SIZE':0.2
                }

    def _runTrain(self, dataFrame):
        embedding_layer=None
        self._tokenizer_initialization(dataFrame, word_vocab_size=self.args['MAX_NUM_WORDS'])
        train_x=self._preprocess(dataFrame, self.w2idx_dict, self.args['MAX_LEN'])

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(dataFrame.label)
        encoded_Y=encoder.transform(dataFrame.label)
        train_y = to_categorical(encoded_Y)
        nb_classes=train_y.shape[1]
        print('Shape of Words Tensor:', train_x.shape)
        print('Build model...')
        model=CNN(
                    embedding_layer=embedding_layer,
                    num_words=self.args['MAX_NUM_WORDS'],
                    embedding_dim=self.args['EMBEDDING_DIM'],
                    max_seq_length=self.args['MAX_LEN'],
                    attention_layer= self.args['ATTENTION'],
                    #dropout_rate= self.args['DROP_OUT'],
                    kernel_sizes=[3],
                    feature_maps=[100],
                    nb_classes=nb_classes,
                    activation='softmax'
                    ).build_model()
            
        model.compile(
                    loss='categorical_crossentropy',
                    optimizer=tf.optimizers.Adam(),
                    metrics=['accuracy']
                    )

        model.summary()
            
        model.fit(
                    train_x, train_y,
                    epochs=self.args['NB_EPOCHS'],
                    batch_size=self.args['BATCH_SIZE']
                    )
        ## save model
        model.save(self.dirModel+'/cnn.h5')
        joblib.dump(encoder, self.dirModel+"labelEncoder.model")
    
    def _tokenizer_initialization(self, dataFrame, word_vocab_size=5000):
        #create tokenizer for our data
        self.w2idx_dict = tf.keras.preprocessing.text.Tokenizer(num_words=word_vocab_size, lower=True, oov_token=True)
        self.w2idx_dict.fit_on_texts(dataFrame.text)
        joblib.dump(self.w2idx_dict, self.dirModel+"word2vec.model")
        
    def _preprocess(self, dataFrame, tokenizer, maxlen=200):
        
        #convert text data to numerical indexes
        train_seqs=tokenizer.texts_to_sequences(dataFrame.text)
        word_index = tokenizer.word_index
        print("unique words : {}".format(len(word_index)))

        return tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=maxlen, padding="post")
