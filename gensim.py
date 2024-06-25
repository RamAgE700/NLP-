import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download NLTK resources if not already downloaded
import nltk
nltk.download('wordnet')

# Take input from the user
text = input("Enter the text to be processed: ")

# Tokenization using Gensim
tokens = simple_preprocess(text, deacc=True)
print(f"\nTokens: {tokens}")

# Lemmatization using NLTK's WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print(f"\nLemmatized Tokens: {lemmatized_tokens}")

# Stemming using NLTK's PorterStemmer
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print(f"\nStemmed Tokens: {stemmed_tokens}")

# Stop word removal using Gensim's built-in stop words
stop_words = STOPWORDS
filtered_tokens = [token for token in tokens if token not in stop_words]
print(f"\nFiltered Tokens (Stop Words Removed): {filtered_tokens}")