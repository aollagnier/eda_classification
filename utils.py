# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
Created on April 22, 2020
@author: Anaïs Ollagnier
This is utils.py contains default option setting function, which is called in main.
"""
import sys, os, optparse, random
import tensorflow as tf
import numpy as np

"""
Add paths
"""
main = os.path.realpath(__file__).split('/')
rootDir = "/".join(main[:len(main)-3])
srcDir = os.path.join(rootDir, 'src')
sys.path.append(srcDir)

def defaultOptions():
    """
    Set default options. Called in Main.py
    """
    parser = optparse.OptionParser(
            usage ='%prog [options] <input data folder> <output data folder>'
            '\n  e.g. (training) python3 main.py -T -t diag -p 0.1 Data/train/ Result/train/'
            '\n       (labeling) python3 main.py -L Data/test/ Result/test/'
            '\n       for more information, type \"python3 main.py\" without --help option'
            )
    parser.add_option('-T', '--Training', dest="T", default=False, action="store_true", help="Classification training")
    parser.add_option('-D', '--Dev', dest="D", default=False, action="store_true", help="Dev Mode")
    parser.add_option('-L', '--Labeling', dest="L", default=False, action="store_true", help="Protest labeling")
    common_opts = optparse.OptionGroup(
            parser, 'Training and labeling options',
            'These options are for both training and labeling'
            )

    common_opts.add_option('-t', '--typetrain', dest="t", default=False, action="store", type='choice', choices=['diag', 'proc'], help="Only when working on eHealth CLEF– Multilingual Information Extraction Tasks")
    common_opts.add_option('-p', '--percent', dest="p", default=0.0, action="store", help="discarde rare classes (%)")
    common_opts.add_option('-b', '--binary', dest="b", default=False, action="store", help="one VS Rest classifier (Multi-Class classification purposes)")
    common_opts.add_option('-a', '--augmentation', dest="a", default='all', action="store", type='choice', choices=['all', 'sr','rd','rs', 'ri'], help="Text augmentation strategy")
    parser.add_option_group(common_opts)

    return parser


def reset_seeds(reset_graph_with_backend=None):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional
        
    np.random.seed(1234)
    random.seed(1234)
    tf.compat.v1.set_random_seed(1234)


