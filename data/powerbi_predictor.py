import pandas as pd
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tashaphyne.stemming import ArabicLightStemmer

# Reconstruct global dependencies
lemmatizer = WordNetLemmatizer()
stemmer = ArabicLightStemmer()

stopwords = [
    "the", "is", "in", "at", "and", "a", "an", "of", "to", "on", "with", "this", "that"
]
list_of_stopwordsAr = [
    "في", "من", "على", "إلى", "و", "عن", "أن", "ما", "مع", "كان", "هذا", "هذه"
]

# Load all pickled components
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("text_cleaner.pkl", "rb") as f:
    text_cleaner = pickle.load(f)

with open("Arabic_text_cleaner.pkl", "rb") as f:
    Arabic_text_cleaner = pickle.load(f)

with open("predict_topic.pkl", "rb") as f:
    predict_topic = pickle.load(f)

# Apply to messages
df['Predicted_Topic'] = df['Messages'].astype(str).apply(lambda x: predict_topic(x, vectorizer, model))
