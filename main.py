# -*- coding: utf-8 -*-
from __future__ import unicode_literals
'''
Created on April 22, 2020 
@author: Anais Ollagnier
This is Main.py that creates classification objects.
'''
import sys
import os
from classification import Classification
from utils import *

if __name__ == '__main__':
    main = os.path.realpath(__file__).split('/')
    rootDir = '/'.join(main[:len(main)-1])
    parser = defaultOptions()
    options, args = parser.parse_args(sys.argv[1:])
    if len(args) < 1 or ((not options.T) and (not options.L) and (not options.D)) :
        print ("--------------------------------------")
        print ("MULTI-LAB EDA: Text Augmentation strategies for Multi-label Classification")
        print ("--------------------------------------")
        print ("Usage in training mode (-T): python3 main.py [options] <input data folder> <output data folder>")
        print ("e.g. training: python3 main.py -T -p 0.1 datasets/final_dataset_v2_to_publish/train/trainP.tsv Result/")
        print ("Usage in training mode (-T):")
        print ("\t python3 main.py [options] <input data folder>")
        print ("\t e.g. training: python3 main.py -T -p 0.1 train_data/")

        print ("Usage in dev mode (-D):")
        print ("\t python3 main.py [options] <input data folder> <gold data folder> <output_folder>")
        print ("\t e.g. dev: python3 main.py -D -p 0.1 /train_data/train.tsv dev_data/gold.tsv Result/")
        
        print ("Usage in labeling mode (-L):")
        print ("\t python3 main.py [options] <input data folder> <output data folder>")
        print ("\t e.g. labeling: python3 main.py -L -p 0.1 train_data/ Result/")

        print ("Options")
        print ("(mode)")
        print ("  -T : --Training")
        print ("\t Data training, default='False'")
        print ("  -D : --Dev")
        print ("\t Data dev, default='False'")
        print ("  -L : --Labeling")
        print ("\t Data labeling, default='False'")
        print ("  -p : --percent")
        print ("\t Percentage of classes to discard, default=0.0")
        print ("  -b : --binary_mode")
        print ("\t Change the data to be trained using One vs Rest mode, default='False'")
        print ("  -a : --augmentation_mode")
        print ("\t Data augmentation strategy, default='all'")

        print ("  -t : --document_type")
        print ("\t Used as a part of the eHealth CLEF– Multilingual Information Extraction Challenge, default=False")


    else:
        classification_process = Classification(options)
        if options.t : model_path= 'model/'+options.t
        else:model_path= 'model/'

        if options.b: dirModel = os.path.join(rootDir, model_path+'/binary_mode/')
        elif options.a: dirModel = os.path.join(rootDir, model_path+'/eda_'+options.a+'/')
        else: dirModel = os.path.join(rootDir, 'model/')
        if not os.path.exists(dirModel): os.makedirs(dirModel)
        if options.T : #training
            classification_process.train(str(args[0]), dirModel)
        elif options.D : #dev
            classification_process.dev(str(args[0]), str(args[1]), str(args[2]), dirModel)
        elif options.L : #labeling
            classification_process.test(str(args[0]), dirModel)
        else:
            print("Please choose training(-T option), dev (-D option) or testing(-L option)")

