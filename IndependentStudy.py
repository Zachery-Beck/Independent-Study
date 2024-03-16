"""Generates df which incudes the TF-IDF"""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk

def is_stopwords_downloaded() -> None:
    """Check to see if Stop words is downloaded if so return already dowladed else adds extra words list and stopwords list"""
    try:
        nltk.data.find('corpora/stopwords')
        print("Stopwords corpus is already downloaded.")
    except LookupError:
        print("Stopwords corpus is not downloaded. Downloading now...")
        nltk.download('stopwords')
        print("Stopwords corpus has been downloaded.")
        custom_stopwords = ['also', 'day', 'make','one','ways','work']
        nltk_stopwords = nltk.corpus.stopwords.words('english')
        nltk_stopwords.extend(custom_stopwords)
        print("Additional values have been added after downloading the stopwords corpus.")
    


def list_txt_files(directory : str) -> list:
    """Return a list of full paths to .txt files in the specified directory."""
    all_files = os.listdir(directory)   
    txt_files = [os.path.join(directory, file) for file in all_files if file.endswith('.txt')]
    return txt_files

def file_to_string(txt_files_list : list) -> str:
    """Read all content from a list of text files and concatenate into a single string."""
    all_content = ""
    for file in txt_files_list:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            all_content += content + " " 
    return all_content

def calculate_tf_idf_simplified(document_string : str) -> pd.DataFrame:
    """Calculate TF-IDF scores for the words in the provided document string and return them as a DataFrame."""
    # Use NLTK's stop words
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=nltk_stopwords)
    
    vectors = vectorizer.fit_transform([document_string])
    feature_names = vectorizer.get_feature_names_out()
    tf_idf_scores = vectors.toarray()[0]
    words_df = pd.DataFrame({'word': feature_names, 'TF-IDF': tf_idf_scores})
    
    return words_df


