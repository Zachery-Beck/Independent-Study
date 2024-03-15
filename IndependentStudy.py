"""TF-IDF/StopWords"""
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

def list_txt_files(directory):
    """ """
    all_files = os.listdir(directory)   
    txt_files = [file for file in all_files if file.endswith('.txt')]
    return txt_files
            

def is_stopwords_downloaded():
    try:
        nltk.data.find('corpora/stopwords')
        print("Stopwords corpus is already downloaded.")
    except LookupError:
        print("Stopwords corpus is not downloaded. Downloading now...")
        nltk.download('stopwords')
        print("Stopwords corpus has been downloaded.")
        
def file_to_list(txt_files_list):
    """ """
    all_words_list = []
    for file in txt_files_list:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            words = content.split()
            all_words_list.append(words)
    return all_words_list


def tif_idf(all_words_list):
    stop_words = list(stopwords.words('english'))

    # Use TfidfVectorizer to compute TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    vectors = vectorizer.fit_transform(all_words_list)

    # Convert vectors to DataFrame
    df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())

    # Add labels
    df['document'] = [all_words_list]

    print(df)