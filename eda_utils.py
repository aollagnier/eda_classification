# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
random.seed(1)
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import text_to_word_sequence

#cleaning up text
import re
def get_only_chars(line):
    clean_line = ""
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("/", " ") #replace slash with spaces
    clean_line = text_to_word_sequence(line, lower=True, filters='“”!"#$%&()*+,-.:;<=>?@[\\]^_`{|}~\t\n')
    #clean_line = [i for i in clean_line if len(i) > 3 and i not in stopwords.words('spanish')]
    return ' '.join(clean_line)

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 

def synonym_replacement(words, n, lang='en'):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words('spanish')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word, lang)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break
        
    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words

def get_synonyms(word, lang):
	synonyms = set()
	for syn in wordnet.synsets(word, lang=lang): 
		for l in syn.lemmas(lang=lang): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words
    
    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    
    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]
    
    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n, lang='en'):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words, lang)
	return new_words

def add_word(new_words, lang):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word, lang)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# Vertical Flip
# Flip a sentence according to a given percentage
########################################################################

def v_flip(words, p):
    words.reverse()
    return words

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, alpha_vf = 0.5, num_aug=9):
    strat = [float(i) for i in [alpha_sr, alpha_ri, alpha_rs, p_rd, alpha_vf]]
    count=strat.count(0.0)
    
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)
   
    augmented_sentences=[]
    if num_words > 0:
        num_new_per_technique = int(num_aug/(len(strat)-count))+1
        #num_new_per_technique = int(num_aug/4)+1
        n_sr = max(1, int(alpha_sr*num_words))
        n_ri = max(1, int(alpha_ri*num_words))
        n_rs = max(1, int(alpha_rs*num_words))
    
        #vf
        if alpha_vf !=0 or alpha_vf !=0.0:
            #print('Vertical Flip...')
            for _ in range(num_new_per_technique):
                a_words = v_flip(words, alpha_vf)
                augmented_sentences.append(' '.join(a_words))

        #sr
        if alpha_sr !=0 or alpha_sr !=0.0:
            #print('Synonym replacement processing...')
            for _ in range(num_new_per_technique):
                a_words = synonym_replacement(words, n_sr, 'spa')
                augmented_sentences.append(' '.join(a_words))
        #ri
        if alpha_ri !=0 or alpha_ri !=0.0:
            #print('Random insertion processing...')
            for _ in range(num_new_per_technique):
                a_words = random_insertion(words, n_ri, 'spa')
                augmented_sentences.append(' '.join(a_words))
        #rs
        if alpha_rs !=0 or alpha_rs !=0.0:
            #print('Random swap processing...')
            for _ in range(num_new_per_technique):
                a_words = random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))
        #rd
        if p_rd !=0 or p_rd !=0.0:
            #print('Random deletion processing...')
            for _ in range(num_new_per_technique):
                a_words = random_deletion(words, p_rd)
                augmented_sentences.append(' '.join(a_words))
    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)
    
    #trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
        
    if len(augmented_sentences) == 0:
        #append the original sentence
        augmented_sentences.append(sentence)
    
    return augmented_sentences
