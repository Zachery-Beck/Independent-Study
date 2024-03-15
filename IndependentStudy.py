"""TF-IDF/StopWords"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Uncomment the following lines if you want to download NLTK's stopwords
# import nltk
# nltk.download('stopwords')

documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'

# Use NLTK's English stopwords
stop_words = list(stopwords.words('english'))

# Use TfidfVectorizer to compute TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words)
vectors = vectorizer.fit_transform([documentA, documentB])

# Convert vectors to DataFrame
df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())

# Add labels
df['document'] = ['Document A', 'Document B']

print(df)