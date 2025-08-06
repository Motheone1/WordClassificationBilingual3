import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tashaphyne.stemming import ArabicLightStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load Arabic stopwords
with open("data/listofStopWords.txt", 'r', encoding='utf-8') as f:
    list_of_stopwordsAr = f.read().split('\n')

stopwordsEN = nltk.corpus.stopwords.words('english')
stemmer = ArabicLightStemmer()
lemmatizer = WordNetLemmatizer()


# ----------------------- Language-Specific Cleaners -----------------------

def Arabic_text_cleaner(sentence):
    try:
        tokens = word_tokenize(str(sentence))
        tokens = [token for token in tokens if token not in list_of_stopwordsAr]
        stemmed_tokens = [stemmer.light_stem(token) for token in tokens if token.strip()]
        return ' '.join(stemmed_tokens)
    except:
        return ""

def text_cleaner(sentence):
    try:
        token = word_tokenize(sentence.lower())
        tokens = [lemmatizer.lemmatize(word) for word in token if word.isalpha() and word not in stopwordsEN]
        return ' '.join(tokens)
    except:
        return ""

def detect_language(text):
    return "arabic" if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', str(text)) else "english"

def clean_message(text):
    return Arabic_text_cleaner(text) if detect_language(text) == "arabic" else text_cleaner(text)

def predict_sentiment(text, vectorizer, model):
    cleaned = clean_message(text)
    vector = vectorizer.transform([cleaned])
    return model.predict(vector)[0]

# ----------------------- Load & Prepare Arabic Dataset -----------------------

arabic_df = pd.read_csv("data/MSAC corpus- software.csv")
arabic_df = arabic_df[['Text', 'Polarity']]
arabic_df = arabic_df.rename(columns={'Text': 'Phrase', 'Polarity': 'Label'})
arabic_df['Language'] = 'arabic'

# Normalize Arabic labels to 'positive' / 'negative'
arabic_df['Label'] = arabic_df['Label'].map({1: 'positive', 0: 'negative'})

# ----------------------- Load & Prepare English Dataset -----------------------

eng_df = pd.read_csv("data/judge-1377884607_tweet_product_company.csv")
eng_df = eng_df[['tweet_text', 'sentiment']]
eng_df = eng_df.rename(columns={'tweet_text': 'Phrase', 'sentiment': 'Label'})
eng_df['Language'] = 'english'

# Make sure sentiment values are already in 'positive'/'negative' format
eng_df = eng_df[eng_df['Label'].isin(['positive', 'negative'])]

# ----------------------- Combine Datasets -----------------------

full_df = pd.concat([arabic_df, eng_df], ignore_index=True)

# Clean all text
full_df['Cleaned'] = full_df['Phrase'].astype(str).apply(clean_message)
full_df = full_df[full_df['Cleaned'].str.strip() != ""]

# ----------------------- Train Model -----------------------

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(full_df['Cleaned'])
y = full_df['Label']

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# ----------------------- Predict on New Unlabeled Messages -----------------------

msg_df = pd.read_csv("data/messages.csv")

msg_df['Sentiment'] = msg_df['Messages'].astype(str).apply(lambda x: predict_sentiment(x, vectorizer, model))

# ----------------------- Export Final Output -----------------------

final_df = msg_df[['Message_Id', 'Messages', 'Sentiment']]
final_df.to_csv("sentiment_output.csv", index=False)

print("✅ Unified sentiment analysis complete. Output saved to 'sentiment_output.csv'")
