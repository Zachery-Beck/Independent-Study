"""Generates df which incudes the TF-IDF"""
import os
import pathlib
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
import re

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
    # file_list= []
    for file in txt_files_list:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            all_content += content
    return all_content

def calculate_tf_idf_simplified(string_list: list, count_dict: dict) -> pd.DataFrame:
    """Calculate TF-IDF scores for the words in the provided document string and return them as a DataFrame."""
    # Use NLTK's stop words
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=nltk_stopwords, input='filename')
    vectors = vectorizer.fit_transform(string_list)
    feature_names = vectorizer.get_feature_names_out()
    
    headers = [pathlib.Path(string).name for string in string_list]
    words_df = pd.DataFrame(vectors.toarray(), columns=feature_names).transpose()
    words_df.columns = headers
    words_df.index.name = "Words"
    words_df['WordCounts'] = words_df.index.map(count_dict).astype(int)
    cols = words_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('WordCounts')))
    words_df = words_df[cols]
    print(words_df.head())
    return words_df

def count_of_words(content: str) -> dict:
    '''Gets count of words in a string as a dict, stripping out punctuation.'''
    words = re.findall(r'\b\w+\b', content.lower())
    word_tuples = map(lambda word: (word, 1), words)
    word_counts = reduce(lambda counts, word_tuple: counts.update({word_tuple[0]: counts.get(word_tuple[0], 0) + word_tuple[1]}) or counts, word_tuples, {})
    return word_counts

def write_to_csv(output : str, fixeddf : pd.DataFrame):
    output_file_path = output
    fixeddf.to_csv(output_file_path)
    return output_file_path