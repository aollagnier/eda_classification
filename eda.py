# -*- coding: utf-8 -*-
from __future__ import unicode_literals

'''
Created on April 21, 2020
@author: Anais Ollagnier

refs: 
        -https://arxiv.org/pdf/1901.11196.pdf
        -https://github.com/jasonwei20/eda_nlp
'''

import random, itertools
from collections import Counter
from random import shuffle
import pandas as pd
from nltk.tokenize import sent_tokenize
import numpy as np

import eda_utils

class EDA:
    def __init__(
            self, strategy
            ):
        self.strategy=strategy
        self.alpha_args ={
                'ALPHA_SR':0.1,
                'ALPHA_RI':0.1,
                'ALPHA_RS':0.1,
                'ALPHA_RD':0.1,
                }


    def _preprocess(self, df, max_sentences=10, ratio=0):

        #if self.strategy == 'all': 
            #df = self._mixing(df, self.alpha_args['ALPHA_MIX'])
        if self.strategy == 'sr':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0
        elif self.strategy == 'ri':
            self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0
        elif self.strategy == 'rs':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0
        elif self.strategy == 'rd':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_SR'] = 0.0, 0.0, 0.0
        majority_class = df['label'].value_counts().max()
        nbre_docs = Counter(df['label'])
        
        labels_list = []
        if ratio != 0:
            for k, v in nbre_docs.items():
                if round(majority_class/v, 1) >= ratio:
                    labels_list.append(k)
        else:
            labels_list = df.label.unique().tolist()
        print(labels_list, len(labels_list)) 
        for label in labels_list:
            aug_sentences = []
            samples_to_generate = majority_class-len(df[df['label'] == label])
            if samples_to_generate != 0:
                filtered_df = df[df['label'] == label]
                docs = filtered_df.text.tolist()
                sentences = sent_tokenize(' '.join(docs))
                for sent in sentences:
                    new_sentences = eda_utils.eda(sent, alpha_sr=self.alpha_args['ALPHA_SR'], alpha_ri=self.alpha_args['ALPHA_RI'], alpha_rs=self.alpha_args['ALPHA_RS'], p_rd=self.alpha_args['ALPHA_RD'], num_aug=9)
                    aug_sentences.append(new_sentences)
            aug_sentences=list(itertools.chain.from_iterable(aug_sentences))
            sentences=list(set(aug_sentences))
            shuffle(sentences)
            i=1
            while i <= (majority_class-nbre_docs[label]):
                new_doc = random.choices(sentences, k=max_sentences)
                if new_doc not in df[df['label'] == label].text.tolist():
                    df = df.append(pd.Series([label, ' '.join(new_doc)], index=df.columns), ignore_index=True)
                    i = i+1

        #if self.strategy == 'all':
            #df = self._mixing(df, self.alpha_args['ALPHA_MIX'])

        '''
        majority_class = Counter(df[df.columns[1]]).most_common(1)[0][1]

        docs = df.text.loc[df[df.columns[1]] == 1].tolist()
        sentences = sent_tokenize(' '.join(docs))
        for sent in sentences:
            new_sentences = eda_bis.eda(sent, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9)
            sentences = sentences + new_sentences

        sentences=list(set(sentences))
        shuffle(sentences)
        i=1
        while i <= (majority_class-len(docs)):
            new_doc = random.choices(sentences, k=max_sentences)
            if new_doc not in df.text.loc[df[df.columns[1]] == 1].tolist():
                df = df.append(pd.Series([' '.join(new_doc), 1], index=df.columns), ignore_index=True)
                i = i+1
        '''
        return df

    def _mixing(self, df, frac=0.1):
        fraction_of_rows = df.sample(frac=frac, random_state=np.random.RandomState())
        for row in fraction_of_rows.itertuples():
            doc_1=eda_utils.get_only_chars(row.text)
            doc_2=df.sample()
            doc_2=eda_utils.get_only_chars(' '.join(doc_2.text.values))
            
            percent_doc_1= random.randint(1,101)
            percent_doc_2= 100-percent_doc_1
            
            words_doc_1, words_doc_2 = doc_1.split(' '), doc_2.split(' ')
            keep_words_doc_1= ((len(words_doc_1) * percent_doc_1) / 100)
            keep_words_doc_2= len(words_doc_2) - ((len(words_doc_2) * percent_doc_2) / 100)
            new_doc = ' '.join(words_doc_1[:int(keep_words_doc_1)]) + ' '.join(words_doc_2[int(keep_words_doc_2):])
            df = df.append(pd.Series([row.label, new_doc], index=df.columns), ignore_index=True)
        return df


