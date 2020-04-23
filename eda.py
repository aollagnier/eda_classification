# -*- coding: utf-8 -*-
from __future__ import unicode_literals

'''
Created on April 21, 2020
@author: Anais Ollagnier

refs: 
        -https://arxiv.org/pdf/1901.11196.pdf
        -https://github.com/jasonwei20/eda_nlp
'''

import random
from collections import Counter
from random import shuffle
import pandas as pd
from nltk.tokenize import sent_tokenize

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
                'ALPHA_RD':0.1
                }


    def _preprocess(self, df, max_sentences=10):
        if self.strategy == 'sr':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0
        elif self.strategy == 'ri':
            self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0
        elif self.strategy == 'rs':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_SR'], self.alpha_args['ALPHA_RD'] = 0.0, 0.0, 0.0
        if self.strategy == 'rd':
            self.alpha_args['ALPHA_RI'], self.alpha_args['ALPHA_RS'], self.alpha_args['ALPHA_SR'] = 0.0, 0.0, 0.0

        majority_class = df['label'].value_counts().max()
        
        labels_list = df.label.unique().tolist()
        for label in labels_list:
            samples_to_generate = majority_class-len(df[df['label'] == label])
            if samples_to_generate != 0:
                filtered_df = df[df['label'] == label]
                docs = filtered_df.text.tolist()
                sentences = sent_tokenize(' '.join(docs))
                for sent in sentences:
                    new_sentences = eda_utils.eda(sent, alpha_sr=self.alpha_args['ALPHA_SR'], alpha_ri=self.alpha_args['ALPHA_RI'], alpha_rs=self.alpha_args['ALPHA_RS'], p_rd=self.alpha_args['ALPHA_RD'], num_aug=9)
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

