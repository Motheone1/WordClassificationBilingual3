import nltk 
import random
import pandas
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB , GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import pickle
from tashaphyne.stemming import ArabicLightStemmer
import re
import csv 
from datetime import datetime , timedelta



# %%
from nltk.tokenize import word_tokenize

# %%
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize , sent_tokenize
import nltk

# Required once
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# %%
nuetral  = open("../data/neutral.txt", 'r', encoding='utf-8').read()
positive = open("../data/positive.txt", 'r', encoding='utf-8').read()
negative = open("../data/negative.txt", 'r', encoding='utf-8').read()
nuetralAR = open("../data/neutral.en.ar.txt", 'r', encoding='utf-8').read()
positiveAR = open("../data/positive.en.ar.txt", 'r', encoding='utf-8').read()
negativeAR = open("../data/negative.en.ar.txt", 'r', encoding='utf-8').read()


nuetral = nuetral.split('\n')
positive = positive.split('\n')
negative = negative.split('\n')
nuetralAR = nuetralAR.split('\n')
positiveAR = positiveAR.split('\n')
negativeAR = negativeAR.split('\n')
stopwordsAR = open("../data/listofStopWords.txt", 'r', encoding='utf-8').read()

# %%
# Arabic Light Stemmer
from tashaphyne.stemming import ArabicLightStemmer
stemmer  = ArabicLightStemmer()

list_of_stopwordsAr = stopwordsAR.split('\n')
punctuation = r'''!()-[]{};:'"\,<>./?@#$%^&*_~،؛؟ـ«»…'''


# %%
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

nuetral0 = [word for word in nuetral if word]
positive0 = [word for word in positive if word] 
negative0 = [word for word in negative if word]
nuetralAR0 = [word for word in nuetralAR if word]
positiveAR0 = [word for word in positiveAR if word]
negativeAR0 = [word for word in negativeAR if word]

print("Positive:", len(positive0))
print("Negative:", len(negative0))
print("Neutral:", len(nuetral0))
print("Positive AR:", len(positiveAR0))
print("Negative AR:", len(negativeAR0))
print("Neutral AR:", len(nuetralAR0))

def text_cleaner(sentence):
    token = word_tokenize(sentence.lower())
    # print("TokeN", token)
    tokens = [lemmatizer.lemmatize(word) for word in token if word.isalpha() and word not in stopwords]
    # print("Tokens", tokens)
    sentence  = ' '.join(tokens)
    # print("After", sentence)
    return sentence




def texter (lst):
    new_lst = []
    for i in range(len(lst)):
        # print("before", lst[i])
        lst[i] = text_cleaner(lst[i])
        if lst[i] == '' or lst[i] == ' ' or lst[i] == '  ' or lst[i] == '   ':
            continue
        print(lst[i])
        # print("after", lst[i])
        if lst[i] != '':
            new_lst.append(lst[i])   
    return new_lst

# %%
def Arabic_text_cleaner(sentence):
    tokens = word_tokenize(sentence)
    for token in range(len(tokens)):
        if tokens[token] in list_of_stopwordsAr:
            tokens[token] = ''             
    stemmed_tokens = [stemmer.light_stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)
    

# %%
def Arabic_texter (lst):
    new_lst = []
    for i in range(len(lst)):
        # print("before", lst[i])
        if lst[i] == '' or lst[i] == ' ' or lst[i] == '  ' or lst[i] == '   ' or lst[i] == '\n' or lst[i] == '\t':
            continue
        lst[i] = Arabic_text_cleaner(lst[i])
        
        # print(lst[i])
        # print("after", lst[i])
        
        new_lst.append(lst[i])   
    return new_lst


nuetral0 = texter(nuetral0)
positive0 = texter(positive0)
negative0 = texter(negative0)
nuetralAR0 = Arabic_texter(nuetralAR0)
positiveAR0 = Arabic_texter(positiveAR0)
negativeAR0 = Arabic_texter(negativeAR0)


documents = []

for w in positive0:
    documents.append((w, 'positive'))
for w in negative0:
    documents.append((w, 'negative'))
for w in nuetral0:
    documents.append((w, 'neutral'))
for w in positiveAR0:
    documents.append((w, 'positive'))
for w in negativeAR0:
    documents.append((w, 'negative'))
for w in nuetralAR0:
    documents.append((w, 'neutral'))