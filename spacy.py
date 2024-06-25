import spacy
from nltk.stem import PorterStemmer

# Load the SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Initialize NLTK's PorterStemmer
stemmer = PorterStemmer()

# Take input from the user
text = input("Enter the text to be processed: ")

# Tokenization using SpaCy
doc = nlp(text)
tokens = [token.text for token in doc]
print(f"\nTokens: {tokens}")

# Lemmatization using SpaCy
lemmatized_tokens = [token.lemma_ for token in doc]
print(f"\nLemmatized Tokens: {lemmatized_tokens}")

# Stemming using NLTK's PorterStemmer
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print(f"\nStemmed Tokens: {stemmed_tokens}")

# Stop word removal (using SpaCy's default stop words list)
filtered_tokens = [token.text for token in doc if not token.is_stop]
print(f"\nFiltered Tokens (Stop Words Removed): {filtered_tokens}")