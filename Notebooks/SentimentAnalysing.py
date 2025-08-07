import pandas as pd
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tashaphyne.stemming import ArabicLightStemmer
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stemmer = ArabicLightStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))

# === File paths ===
base_dir = r"C:\Users\Mohammed\Documents\work_stuff\training\TopicClassifier"
csv_path = rf"{base_dir}\cosmos_data211_predicted.csv"
output_path = rf"{base_dir}\cosmos_data211_with_sentiment.csv"
stopwords_path = rf"{base_dir}\data\listofStopWords.txt"
model_path = rf"{base_dir}\Pickles\model2.pkl"
vectorizer_path = rf"{base_dir}\Pickles\vectorizer2.pkl"

# Load Arabic stopwords
with open(stopwords_path, 'r', encoding='utf-8') as f:
    list_of_stopwordsAr = set(f.read().splitlines())

# Load model and vectorizer
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# English text cleaning
def text_cleaner(sentence):
    token = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(word) for word in token if word.isalpha() and word not in stopwords]
    return ' '.join(tokens)

# Arabic text cleaning
def Arabic_text_cleaner(sentence):
    tokens = word_tokenize(sentence)
    filtered_tokens = [token for token in tokens if token not in list_of_stopwordsAr and token.strip()]
    stemmed_tokens = [stemmer.light_stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)

# Predict sentiment
def predict_sentiment(text):
    text = str(text)
    if bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text)):
        cleaned_text = Arabic_text_cleaner(text)
    else:
        cleaned_text = text_cleaner(text)
    vector = vectorizer.transform([cleaned_text])
    return model.predict(vector)[0]

# Load dataset
df = pd.read_csv(csv_path)

# Verify 'Content' column exists
if 'content' not in df.columns:
    raise ValueError("❌ 'Content' column not found in the CSV.")

# Apply sentiment prediction
df['Sentiment_Analysis'] = df['content'].astype(str).apply(predict_sentiment)

# Save the updated CSV
df.to_csv(output_path, index=False)
print(f"✅ Done! Saved to: {output_path}")
