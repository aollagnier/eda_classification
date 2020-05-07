# -*- coding: utf-8 -*-
from __future__ import unicode_literals

'''
Created on April 30, 2020
@author: Anais Ollagnier

'''

from sentence_transformers.sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformers import models
import re, joblib, os, itertools, random
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize
from transformers import *

from model import Model
import utils, eda_utils

class DataGenerator(Sequence):
    def __init__(self, docs, labels, dirModel, options, n_classes=10,# data settings
            vocab_size=10000, max_len=24, embedding = None, #vocabulary settings
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
        # Use BERT for mapping tokens to embeddings
        #word_embedding_model = models.Transformer('bert-base-multilingual-cased') 
        #print(word_embedding_model) 
        
        self.embedding_model={
                'transfo': (TransfoXLTokenizer, 'transfo-xl-wt103', TransfoXLConfig, TFTransfoXLModel),
                'ctrl': (CTRLTokenizer, 'ctrl', CTRLConfig, TFCTRLModel),
                'electra': (ElectraTokenizer, 'google/electra-small-discriminator', ElectraConfig, TFElectraModel),
                't5': (T5Tokenizer, 't5-base', T5Config, TFT5Model),
                'bert': (BertTokenizer, 'bert-base-multilingual-uncased', BertConfig, TFBertModel),
                'gpt2': (GPT2Tokenizer, 'gpt2', GPT2Config, TFGPT2Model),
                'xlnet': (XLNetTokenizer, 'xlnet-base-cased', XLNetConfig, TFXLNetModel),
                'roberta': (RobertaTokenizer, 'roberta-base', RobertaConfig, TFRobertaModel),
                'distil': (DistilBertTokenizer, 'distilbert-base-cased', DistilBertConfig, TFDistilBertModel),
                'albert': (AlbertTokenizer, 'albert-base-v2', AlbertConfig, TFAlbertModel)
                }
        
        self.embedding = embedding
        if self.embedding is not None:
            ntokenizer, self.nmodel, self.config, self.model_class = self.embedding_model[self.embedding][0], self.embedding_model[self.embedding][1], self.embedding_model[self.embedding][2], self.embedding_model[self.embedding][3]
            self.tokenizer=ntokenizer.from_pretrained(self.nmodel)
        
        self.docs = docs
        self.labels = labels
        self.n_classes = n_classes
        self.embedding = embedding
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
                'ALPHA_VF':0.5
                }
        
        if self.strategy == 'sr':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_RD'], self.alpha_args['ALPHA_VF'] = 0.0, 0.0, 0.0, 0.0
        elif self.strategy == 'ri':
            self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_RD'], self.alpha_args['ALPHA_VF'] = 0.0, 0.0, 0.0, 0.0
        elif self.strategy == 'rs':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_RD'], self.alpha_args['ALPHA_VF'] = 0.0, 0.0, 0.0, 0.0
        elif self.strategy == 'rd':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_VF'] = 0.0, 0.0, 0.0, 0.0
        elif self.strategy == 'vf':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0, 0.0
        
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

    def __create_encodings(self, data):
        #"""Build the encodings for the provided data"""
        #self.docs = self.doc_vocab.texts_to_sequences(self.docs)
        words_encoder= self.doc_vocab.texts_to_sequences(data)
        batch_words = tf.keras.preprocessing.sequence.pad_sequences(words_encoder, maxlen=self.max_len, padding='post', truncating='post')
        batch_docs = self.__sent2vec(data, self.doc_vocab)
        #print([np.vstack(batch_docs),
                      #np.vstack(batch_words)])
        return [batch_words, batch_docs]

    def __sent2vec(self, X, tokenizer, sent_max_len=20, max_sentences=10):
        data_feature = np.zeros((len(X), max_sentences, sent_max_len), dtype='int32')
        for i, sentences in enumerate(X):
            sentences = sent_tokenize(sentences)
            sentences = sentences[:max_sentences]
            w= self.doc_vocab.texts_to_sequences(sentences)
            w= tf.keras.preprocessing.sequence.pad_sequences(w, maxlen=sent_max_len, padding="post", truncating='post')
            if len(w) < max_sentences:
                w = np.pad(w, ((0,max_sentences-len(w)),(0,0)), mode='constant')
            data_feature[i] = w
        return data_feature
    
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        
        return int(np.floor(len(self.docs) / self.batch_size))

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
        y = to_categorical(encoded_Y, num_classes=self.n_classes)
        # Generate data
        if self.options.a:
            X = self.__data_augmentation(X)
    
        if self.embedding:

            inps = [self.tokenizer.encode_plus(t, max_length=self.max_len, pad_to_max_length=True, add_special_tokens=True) for t in X]
            X = np.array([a['input_ids'] for a in inps])
            
            #config=self.config.from_pretrained(self.mname)
            #config.output_hidden_states = False
            #model=self.model_class.from_pretrained(self.mname, config=config)
            #X,y=self.mixup_data(model([input_ids])[0], y)
        else:
            X= self.doc_vocab.texts_to_sequences(X)
            X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.max_len, padding="post")
            #self.__embedding_matrix()
            #X= self.__create_encodings(X)
        
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.docs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_augmentation(self, list_doc_temp):
        'Returns augmented data with batch_size docs' # X : (n_samples, max_len)
        
        # Generate new samples
        X=[]
        for doc in list_doc_temp:
            sent_tokenize_list = sent_tokenize(doc)
            new_doc = [random.choice(eda_utils.eda(sent, 
                alpha_sr=self.alpha_args['ALPHA_SR'], # synonyms
                alpha_ri=self.alpha_args['ALPHA_RI'], # random insertion
                alpha_rs=self.alpha_args['ALPHA_RS'], # random swap
                p_rd=self.alpha_args['ALPHA_RD'], #random deletion
                alpha_vf=self.alpha_args['ALPHA_VF'], #vertical flip
                num_aug=1)) for sent in sent_tokenize_list]
            X.append(' '.join(new_doc))
        return X
    
    def mixup_data(self, x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        sample_size = x.shape[0]
        index_array = np.arange(sample_size)
        np.random.shuffle(index_array)
        X = np.empty((self.batch_size, # n_docs
                      self.max_len, # dimension w.r.t. x
                      x.shape[2] # dimension w.r.t. bert embedding
                      ))
        y = []
        
        for i in range(self.batch_size):
            #store mixed doc
            X[i] = lam * x[i] + (1 - lam) * x[index_array.tolist()[i]]
        #mixed_x = lam * x + (1 - lam) * x[index_array]
        #mixed_y = (lam * y) + ((1 - lam) * y[index_array])
        
        #print((1 - lam) * y[index_array])
        #print((lam * y).shape,((1 - lam) * y[index_array]).shape)
        return X, y
     
    
    '''
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
