import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Example text
text = "Hello, world! This is an example sentence. Let's see how it gets tokenized, lemmatized, and stemmed."

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Word Tokenization
words = word_tokenize(text)
print("Words:", words)

# Initialize Lemmatizer and Stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Get stop words
stop_words = set(stopwords.words('english'))

# Remove Stop Words, Lemmatize and Stem
filtered_words = []
lemmatized_words = []
stemmed_words = []

for word in words:
    # Remove punctuation and convert to lowercase
    if word.isalpha():
        word_lower = word.lower()
        if word_lower not in stop_words:
            filtered_words.append(word_lower)
            lemmatized_words.append(lemmatizer.lemmatize(word_lower))
            stemmed_words.append(stemmer.stem(word_lower))

print("Filtered Words (Stop Words Removed):", filtered_words)
print("Lemmatized Words:", lemmatized_words)
print("Stemmed Words:", stemmed_words)