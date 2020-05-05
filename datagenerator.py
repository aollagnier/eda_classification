# -*- coding: utf-8 -*-
from __future__ import unicode_literals

'''
Created on April 30, 2020
@author: Anais Ollagnier

'''
import re, joblib, os, itertools, random
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize

from model import Model
import utils, eda_utils

class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, n_classes=10, shuffle=True):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.n_classes = n_classes

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        print(self.X_train.shape)

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(self.y_train)
        encoded_Y=encoder.transform(self.y_train)
        self.y_train = to_categorical(encoded_Y)
        print(type(self.y_train))
        #_, h, w, c = self.X_train.shape
        _, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        print('------------', X1)
        X2 = self.X_train[batch_ids[self.batch_size:]]
        print('************', X2)
        X = X1 * X_l + X2 * (1 - X_l)
        
        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            print('yoooooooooooooooo')
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)
        
        print(batch_ids)
        X = self.X_train[batch_ids[:self.batch_size]]
        y = self.y_train[batch_ids[:self.batch_size]]
        print(X.shape)
        print(y.shape)
        return X, y

class DataGenerator(Sequence):
    def __init__(self, docs, labels, dirModel, options, n_classes=10,# data settings
            vocab_size=10000, max_len=24, #vocabulary settings
            batch_size=32,  shuffle=True #augmentation settings
            ):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        
        utils.reset_seeds()

        self.docs = docs
        self.labels = labels
        self.n_classes = n_classes
        self.dirModel = dirModel
        self.options = options
        self.vocab_size= vocab_size
        self.max_len = max_len

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.strategy=self.options.a
        self.alpha_args ={
                'ALPHA_SR':0.1,
                'ALPHA_RI':0.1,
                'ALPHA_RS':0.1,
                'ALPHA_RD':0.1,
                }
        
        self.doc_vocab = None
        self.build_vocab()

        self.on_epoch_end()

    
    def build_vocab(self):
        """Build the vocabulary"""
        
        doc_vocab = set()
        
        docs = [eda_utils.get_only_chars(row) for row in self.docs]

        #create tokenizer for our data
        self.doc_vocab = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size, lower=True, oov_token=True)
        self.doc_vocab.fit_on_texts(docs)
        joblib.dump(self.doc_vocab, os.path.join(self.dirModel, "word2vec.model"))

        word_index = self.doc_vocab.word_index
        print("unique words : {}".format(len(word_index)))

    #def create_encodings(self):
        #"""Build the encodings for the provided data"""
        #self.docs = self.doc_vocab.texts_to_sequences(self.docs)
    
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        
        return int(np.floor(len(self.docs) / self.batch_size))
    '''
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        #list_IDs_temp = [self.features[k] for k in indexes]
        #print(list_IDs_temp)
        # Generate data
        #X, y = self.__data_generation(list_IDs_temp)
        #print(list_IDs_temp.shape)
        #data = np.random.random((1000, 100))
        #labels = np.random.randint(10, size=(1000, 1))

        # Convert labels to categorical one-hot encoding
        encoder = LabelEncoder()
        encoder.fit(self.labels)
        encoded_Y=encoder.transform(self.labels)
        one_hot_labels = to_categorical(encoded_Y, num_classes=self.n_classes)
        print(one_hot_labels)
        #X, y = self.__data_generation(list_IDs_temp)
        
        #if self.to_fit:
            #y = self._generate_y(list_IDs_temp)
            #return X, y
        #else:
            #return X
        return X,y
        '''

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Convert labels to categorical one-hot encoding
        encoder = LabelEncoder()
        encoder.fit(self.labels)
        joblib.dump(encoder, self.dirModel+"/labelEncoder.model")
        
        # Find list of IDs
        X = [self.docs[k] for k in indexes]
        y = [self.labels[k] for k in indexes]
        encoded_Y=encoder.transform(y)

        # Generate data
        if self.options.a:
            X = self.__data_augmentation(X)
    
        X= self.doc_vocab.texts_to_sequences(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.max_len, padding="post")
        
        return X, to_categorical(encoded_Y, num_classes=self.n_classes)

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.docs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_augmentation(self, list_doc_temp):
        'Returns augmented data with batch_size docs' # X : (n_samples, max_len)
        # Initialization
        if self.strategy == 'sr':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0
        elif self.strategy == 'ri':
            self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0
        elif self.strategy == 'rs':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0
        elif self.strategy == 'rd':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_SR'] = 0.0, 0.0, 0.0
        
        # Generate new samples
        X=[]
        for doc in list_doc_temp:
            sent_tokenize_list = sent_tokenize(doc)
            new_doc = [random.choice(eda_utils.eda(sent, alpha_sr=self.alpha_args['ALPHA_SR'], alpha_ri=self.alpha_args['ALPHA_RI'], alpha_rs=self.alpha_args['ALPHA_RS'], p_rd=self.alpha_args['ALPHA_RD'], num_aug=1)) for sent in sent_tokenize_list]
            X.append(' '.join(new_doc))
        '''
        X = np.empty((self.batch_size, 
            self.max_len))# n_enzymes
                      #self.v_size, # dimension w.r.t. x
                      #self.v_size, # dimension w.r.t. y
                      #self.v_size, # dimension w.r.t. z
                      #self.n_channels)) # n_channels
        y = np.empty((self.batch_size), dtype=int)

        # Computations
        for i in range(self.batch_size):
            print(self.labels[self.indexes[i]])
            # Store class
            y[i] = self.labels[self.indexes[i]]
            '''
        return X
    ''' 
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.batch_size,))
        y = np.empty((self.batch_size), dtype=int)
        
        for text, label in enumerate():
            # Store sample
            X[i,] = self.features 

            # Store class
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def mixup_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        print(x, x.shape) 
        sample_size = x.shape[0]
        index_array = np.arange(sample_size)
        np.random.shuffle(index_array)
        print(sample_size, index_array)
        print( x[index_array])
        mixed_x = lam * x + (1 - lam) * x[index_array]
        mixed_y = (lam * y) + ((1 - lam) * y[index_array])
        
        print((1 - lam) * y[index_array])
        print((lam * y).shape,((1 - lam) * y[index_array]).shape)
        return mixed_x, mixed_y
    
    def make_batches(self, size, batch_size):
        nb_batch = int(np.ceil(size/float(batch_size)))
        return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]
    
    def batch_generator(self, X,y,batch_size=128,shuffle=True,mixup=False):
        sample_size = X.shape[0]
        index_array = np.arange(sample_size)
        
        while 1:
            if shuffle:
                np.random.shuffle(index_array)
            batches = self.make_batches(sample_size, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                X_batch = X[batch_ids]
                y_batch = y[batch_ids]
                
                if mixup:
                    X_batch,y_batch = self.mixup_data(X_batch,y_batch,alpha=1.0)
                    print(X_batch.shape,y_batch.shape)
                    
                yield X_batch,y_batch
       '''   
