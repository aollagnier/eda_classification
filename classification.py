# -*- coding: utf-8 -*-
from __future__ import unicode_literals
'''
Created on April 22, 2020
@author: Anais Ollagnier

Train, dev and test functions

'''

import os, gc, pickle, operator, itertools, joblib, csv
import tensorflow as tf
from datetime import datetime
from tempfile import mkdtemp
from  itertools import chain
from collections import Counter
import numpy as np 

from keras2vec.keras2vec import Keras2Vec
from keras2vec.document import Document
from corpus import Corpus
from model import Model, OneVSRest
from eda import EDA
from datagenerator import DataGenerator

class Classification(object):
    def __init__(self, options={}): 
        main = os.path.realpath(__file__).split('/')
        self.rootDir = '/'.join(main[:len(main)-1])
        self.options = options

        self.runs = 1

    def train(self, dirCorpus, dirModel):
        # Usage: python3 main.py -T -t proc -p 0.1 data/final_dataset_v2_to_publish/train/trainP.tsv
        
        print("data processing...")
        corpus = Corpus(dirCorpus, dirModel, self.options)
        data = corpus._preprocess(self.options.p)
        labels = data.label.unique().tolist()

        data = DataGenerator(data.text, data.label.values, dirModel, self.options, max_len=396, n_classes= len(labels))

        print("Training...")
        if self.options.b: model = OneVSRest(dirModel, self.options)
        else: model = Model(dirModel, self.options)
        
        current_time_bf=datetime.now()
        model._runTrain(data)

        current_time_af=datetime.now()
        c = current_time_af - current_time_bf
        print("Elapsed time =", c.total_seconds())
        print('Model saved in {}'.format(dirModel)) 
        gc.collect()

    def dev(self, dirTrainCorpus, dirDevCorpus, dirResult, dirModel):
        # usage: python3 main.py -D -t proc -p 0.1 data/final_dataset_v2_to_publish/train/trainP.tsv data/final_dataset_v2_to_publish/dev/devP.tsv Result/

        if dirResult == '' : dirResult = os.path.join(self.rootDir, 'Result')
        if not os.path.exists(dirResult): os.makedirs(dirResult)
        
        print("Data processing...")
        corpus = Corpus(dirTrainCorpus, dirModel, self.options)
        results=[]
        
        for i in range(self.runs):
            dirResult = mkdtemp(dir = dirResult) + '/'
            data = corpus._preprocess(self.options.p)
            labels = data.label.unique().tolist()
            
            print('Running iteration %i/%i' % (i+1, self.runs))
            #data = DataGenerator(data.text, data.label.values, dirModel, self.options, max_len=396, n_classes= len(labels), val_gen=True)
            data = DataGenerator(data.text, data.label.values, dirModel, self.options, batch_size=8, max_len=396, embedding = 'bert', n_classes= len(labels))
            #data = DataGenerator(data.text, data.label.values, dirModel, self.options, batch_size=50, max_len=396, n_classes= len(labels))
            print("Training...")
            if self.options.b: model = OneVSRest(dirModel, self.options)
            else: model = Model(dirModel, self.options)
            current_time_bf=datetime.now()
            model._runTrain(data)
            '''
            model= self.buildcnn()
            model.compile(
                    loss='categorical_crossentropy',
                    optimizer=tf.optimizers.Adam(),
                    metrics=['accuracy']
                    )
            model.summary()
            
            model.fit(data, epochs=10,
                #steps_per_epoch=np.ceil(float(len(generator.docs)) / float(self.args['BATCH_SIZE'])),
                steps_per_epoch=len(data),
                verbose=1)
            '''
            current_time_af=datetime.now()
            c = current_time_af - current_time_bf
            print("Elapsed time =", c.total_seconds())
            gc.collect()
            
            print("Dev data processing...")
            data = corpus._preprocessDev(dirDevCorpus)
            print(data.head())
            with open(dirModel+"/word2vec.model", 'rb') as handle:
                loaded_tokenizer = pickle.load(handle)
        
            if self.options.b:
                labels_list = chain(*df.label)
                label_keys=labels_list.keys()
                print(labels_list, len(labels_list))

                preds = pd.DataFrame()
                for label in label_keys:
                    print('Working with {}...'.format(dirModel+label+'.h5'))
                    loaded_model = tf.keras.models.load_model(dirModel+label+'.h5')
                    preds[label] = [Codiesp._threshold(i, 0.1) for i in loaded_model.predict(dev_seqs)]
                    tf.keras.backend.clear_session()
                    del loaded_model
                    gc.collect()
                preds=preds.values.tolist()
                preds= [[i for i, pred in enumerate(example) if pred == 1] for example in preds]
                preds= [[uniq_labels[idx] for idx in example] for example in preds]

            else:
                with open(dirModel+"/labelEncoder.model", 'rb') as le:
                    lb = joblib.load(le)
                dev_seqs=model._preprocess(data, loaded_tokenizer, maxlen=396)
                print('Shape of data tensor:', dev_seqs.shape)
        
                loaded_model = tf.keras.models.load_model(dirModel+'/cnn.h5')
                print('Model used: {}'.format(dirModel+'/cnn.h5'))
                preds = loaded_model.predict(dev_seqs)
                #preds= [[i for i, pred in enumerate(example) if Classification._threshold(pred, 0.5) == 1] for example in preds]
                preds= [[(i, pred) for i, pred in enumerate(example) if Classification._threshold(pred, 0.5) == 1] for example in preds]
                preds= [sorted(example, key=lambda x: x[1], reverse=True) for example in preds]
                preds= [[pred[0] for i, pred in enumerate(example)] for example in preds]
                preds = [list(lb.inverse_transform(i)) for i in preds]

            data['prediction'] = preds
            with open(dirResult+'output'+self.options.t+'.tsv', 'wt') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                for row in data[['id', 'prediction']].itertuples():
                    code = list(row.prediction)
                    for c in code:
                        tsv_writer.writerow([row.id, c.upper()])
        
        
            predictions = dirResult+'output'+self.options.t+'.tsv'
        
            if self.options.t:
        
                if self.options.t == 'diag':
                    gold_file = 'data/final_dataset_v2_to_publish/dev/devD.tsv'
                    codes_file = 'data/codiesp-D_codes.tsv'
                else:
                    gold_file = 'data/final_dataset_v2_to_publish/dev/devP.tsv'
                    codes_file = 'data/codiesp-P_codes.tsv'
            
                command = 'python CodiEsp-Evaluation-Script/codiespD_P_evaluation.py -g '+gold_file+' -p '+predictions+' -c '+codes_file
                results.append(command)
                print(command)
        print(results)

    def buildcnn(self):
        # Define token ids as inputs
        word_inputs = tf.keras.layers.Input(shape=(396, 768), name='word_inputs', dtype='float32')
        doc_inputs = tf.keras.layers.Input(shape=(10, 20), name='doc_inputs', dtype='int32')
        
        #embedding_layer = tf.keras.layers.Embedding(
                #input_dim=self.sent_max_len,
                #input_dim=10000,
                #output_dim=300,
                #input_length=20,
                #weights=None, trainable=True,
                #name="word_embedding"
            #)(word_inputs)
        x=tf.keras.layers.SpatialDropout1D(0.0)(word_inputs)
        conv=tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', name="input")(x)
        #conv = tf.keras.layers.Conv2D(32, kernel_size=(3, 300), kernel_initializer='normal', activation='elu', input_shape=(10, 20, 300))(embedding_layer)
        x = tf.keras.layers.MaxPooling1D()(conv)
        x = tf.keras.layers.LSTM(100)(x)
        x = tf.keras.layers.Flatten()(x)
        dense = tf.keras.layers.Dense(64, activation='relu')(x)

        # Final output
        outputs = tf.keras.layers.Dense(563, activation='softmax', name='outputs')(dense)

        # Compile model
        model = tf.keras.Model(inputs=[word_inputs, doc_inputs], outputs=[outputs])
        return model

    
    def test(self, dirTestCorpus, dirModel):
        # usage: python3 main.py -L -t proc data/final_dataset_v2_to_publish/test/text_files/ model/proc/cnn.h5

        dirResult = ''
        if dirResult == '' : dirResult = os.path.join(self.rootDir, 'Result')
        if not os.path.exists(dirResult): os.makedirs(dirResult)
        dirResult = mkdtemp(dir = dirResult) + '/'

        print("Data processing...")
        corpus = Corpus(dirTestCorpus, dirModel, self.options)
        data = corpus._preprocessTest(dirTestCorpus)
        print(data.head())
        
        model = Model(dirModel, self.options)
        with open(dirModel+"/word2vec.model", 'rb') as handle:
            loaded_tokenizer = pickle.load(handle)

        with open(dirModel+"/labelEncoder.model", 'rb') as le:
            lb = joblib.load(le)
            test_seqs=model._preprocess(data, loaded_tokenizer, maxlen=396)
            print('Shape of data tensor:', test_seqs.shape)
            
            loaded_model = tf.keras.models.load_model(dirModel+'cnn.h5')
            print('Model used: {}'.format(dirModel+'cnn.h5'))
            preds = loaded_model.predict(test_seqs)
            #preds= [[i for i, pred in enumerate(example) if Classification._threshold(pred, 0.5) == 1] for example in preds]
            preds= [[(i, pred) for i, pred in enumerate(example) if Classification._threshold(pred, 0.5) == 1] for example in preds]
            preds= [sorted(example, key=lambda x: x[1], reverse=True) for example in preds]
            preds= [[pred[0] for i, pred in enumerate(example)] for example in preds]
            preds = [list(lb.inverse_transform(i)) for i in preds]
        data['prediction'] = preds
        with open(dirResult+'output'+self.options.t+'.tsv', 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for row in data[['id', 'prediction']].itertuples():
                code = list(row.prediction)
                for c in code:
                     tsv_writer.writerow([row.id, c.upper()])

        print('Result file: {}'.format(dirResult+'output'+self.options.t+'.tsv'))

    @staticmethod
    def _threshold(x, threshold):
        if x >= threshold:
            return 1
        return 0

