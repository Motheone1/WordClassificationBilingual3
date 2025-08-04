import pandas as pd
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tashaphyne.stemming import ArabicLightStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('wordnet')    

def text_cleaner(sentence):
    token = word_tokenize(sentence.lower())
    # print("TokeN", token)
    tokens = [lemmatizer.lemmatize(word) for word in token if word.isalpha() and word not in stopwords]
    # print("Tokens", tokens)
    sentence  = ' '.join(tokens)
    # print("After", sentence)
    return sentence


lemmatizer = WordNetLemmatizer()
stemmer = ArabicLightStemmer()

stopwordsAR = open("data/listofStopWords.txt", 'r', encoding='utf-8').read()
list_of_stopwordsAr = stopwordsAR.split('\n')
punctuation = r'''!()-[]{};:'"\,<>./?@#$%^&*_~،؛؟ـ«»…'''
stopwords = nltk.corpus.stopwords.words('english')




def Arabic_text_cleaner(sentence):
    tokens = word_tokenize(sentence)
    for token in range(len(tokens)):
        if tokens[token] in list_of_stopwordsAr:
            tokens[token] = ''             
    stemmed_tokens = [stemmer.light_stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


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




def predict_topic(text, vectorizer, model):
    if bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text)):
        print("Arabic text detected, using Arabic text cleaner.")
        print("the text is:", text )
        cleaned_text = Arabic_text_cleaner(text)
    else:
        print("English text detected, using English text cleaner.")
        cleaned_text = text_cleaner(text)
        print("the text is:", text )
    cleaned_text = [cleaned_text]  # Ensure cleaned_text is a list for vectorization
     # Wrap in a list for vectorization
    print("Cleaned Text:", cleaned_text)  # Print cleaned text for debugging
    tfidf_vector = vectorizer.transform(cleaned_text)
    prediction = model.predict(tfidf_vector)

    return prediction[0]