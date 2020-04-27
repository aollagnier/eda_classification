# -*- coding: utf-8 -*-
from __future__ import unicode_literals
'''
Created on March 27, 2020
@author: Anais Ollagnier

Data cleaning processes
'''
import os, glob, unidecode
import pandas as pd
from collections import Counter
from  itertools import chain

from tensorflow.keras.preprocessing.text import text_to_word_sequence

class Corpus(object):
    def __init__(self, data, dirModel, options):
        self.data = data
        self.dirModel = dirModel
        self.options = options

    def _preprocess(self, percentage):
        files= [os.path.splitext(filename)[0] for filename in os.listdir(self.dirModel)]
        df=pd.read_csv(self.data, sep='\t', names=["id", "label"])
        uniq_labels = df.label.unique().tolist()
        
        diagnostic_content=Corpus._getFiles('data/final_dataset_v2_to_publish/train/text_files/')
        if self.options.b : df = df.groupby('id')['label'].apply(list).reset_index(name='label')
        
        for row in df.itertuples():
            if row.id in diagnostic_content:
                df.at[row.Index, 'text'] = diagnostic_content[row.id]
        del diagnostic_content

        if percentage !=0.0 or percentage!=0:
            df = Corpus._get_refine_data(df, uniq_labels, percentage, binary=self.options.b)
        df = df.drop(['id'], axis=1)
        return df

    def _preprocessDev(self, data):
        diagnostic_content=Corpus._getFiles('data/final_dataset_v2_to_publish/dev/text_files/')
        df= pd.read_csv(data, sep='\t', names=["id", "label"])
        df = df.drop_duplicates(['id'])
        for row in df.itertuples():
            if row.id in diagnostic_content:
                df.at[row.Index, 'text'] = diagnostic_content[row.id]
        del diagnostic_content
        return df

    def _preprocessTest(self, data):
        diagnostic_content=Corpus._getFiles(data)
        df = pd.DataFrame(list(diagnostic_content.items()), columns = ['id', 'text']) 
        del diagnostic_content
        return df

    @staticmethod
    def _get_refine_data(df, uniq_labels, percentage, binary=False):
        if binary : labels=chain(*df.label) 
        else : labels=df.label

        counter=Counter(labels)
        labels_counter = list(counter.items())
            
        refined_labels = [labels_counter[i][0] for i in range(len(labels_counter)) if (labels_counter[i][1]*100/sum(counter.values())) > float(percentage)]
        print('After discarding {}, it remains {} unique classes'.format(percentage, len(refined_labels)))
        
        if binary:
            df.label = [[i for i in L if i in refined_labels] for L in df.label]
            df=df.reindex(columns=[*df.columns.tolist(), *refined_labels], fill_value=0)
            for i, row in df.iterrows():
                for c in row['label']:
                    df.at[i,c]=1
            df = df.drop(['label'], axis=1)

        else : df = df[df['label'].isin(refined_labels)]
        return df
    
    @staticmethod
    def _getFiles(path):

        diagnostic_content={}

        for filepath in glob.glob(os.path.join(path, '*.txt')):
            with open(filepath) as f:
                content = f.read()
                content =  unidecode.unidecode(content)
                root = os.path.basename(filepath)
                #filtered_content = text_to_word_sequence(' '.join(content.split()), filters='“”!"#$%&()*+,-.:;<=>?@[\\]^_`{|}~\t\n')
                #content=[i for i in filtered_content if len(i) > 2 and i not in stopwords.words('spanish')]
                #diagnostic_content[root.rsplit('.', 1)[0]]=' '.join(content)
                diagnostic_content[root.rsplit('.', 1)[0]]=content
        return diagnostic_content

