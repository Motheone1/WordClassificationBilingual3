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
# Load the trained model and vectorizer
with open("Notebooks/model2.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("Notebooks/vectorizer2.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define English and Arabic text cleaners
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
stemmer = ArabicLightStemmer()

with open("data/listofStopWords.txt", 'r', encoding='utf-8') as f:
    list_of_stopwordsAr = f.read().splitlines()

def text_cleaner(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stopwords]
    return ' '.join(tokens)

def Arabic_text_cleaner(sentence):
    tokens = word_tokenize(sentence)
    filtered = [token for token in tokens if token not in list_of_stopwordsAr]
    stemmed_tokens = [stemmer.light_stem(token) for token in filtered]
    return ' '.join(stemmed_tokens)

# Prediction function
def predict_sentiment(text):
    if bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text)):
        cleaned = Arabic_text_cleaner(text)
    else:
        cleaned = text_cleaner(text)

    tfidf_vector = vectorizer.transform([cleaned])
    return model.predict(tfidf_vector)[0]

# Load the CSV
df = pd.read_csv("data/messages.csv", encoding='utf-8')

# Predict sentiment for each message
df['Sentiment'] = df['Messages'].apply(predict_sentiment)

# Output new CSV
df[['Message_Id', 'Messages', 'Sentiment']].to_csv("messages_with_sentiment.csv", index=False, encoding='utf-8')

print("Sentiment analysis completed. File saved as 'messages_with_sentiment.csv'")
