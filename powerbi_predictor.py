import pandas as pd
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tashaphyne.stemming import ArabicLightStemmer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load vectorizer and model (relative paths for portability)
with open(r"D:\-USERS\Mohammed\Channels\Training\WordClassificationBilingual3\vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(r"D:\-USERS\Mohammed\Channels\Training\WordClassificationBilingual3\model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Arabic stopwords
with open(r"D:\-USERS\Mohammed\Channels\Training\WordClassificationBilingual3\data\listofStopWords.txt", "r", encoding="utf-8") as f:
    list_of_stopwordsAr = f.read().splitlines()

# Set up tools
lemmatizer = WordNetLemmatizer()
stemmer = ArabicLightStemmer()
english_stopwords = stopwords.words("english")
punctuation = r'''!()-[]{};:'"\,<>./?@#$%^&*_~،؛؟ـ«»…'''

# Text cleaning functions
def text_cleaner(sentence):
    tokens = word_tokenize(sentence.lower())
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in english_stopwords]
    return " ".join(cleaned)

def Arabic_text_cleaner(sentence):
    tokens = word_tokenize(sentence)
    tokens = ["" if t in list_of_stopwordsAr else t for t in tokens]
    stemmed = [stemmer.light_stem(t) for t in tokens if t]
    return " ".join(stemmed)

# Prediction function
def predict_topic(text, vectorizer, model):
    if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text):  # Arabic regex
        cleaned_text = Arabic_text_cleaner(text)
    else:
        cleaned_text = text_cleaner(text)
    tfidf_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_vector)
    return prediction[0]

# Power BI input dataframe is called 'dataset'
dataset['Predicted_Topic'] = dataset['Messages'].astype(str).apply(
    lambda x: predict_topic(x, vectorizer, model)
)

# Output to Power BI
dataset
