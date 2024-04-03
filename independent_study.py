"""Generates DataFrame which includes the TF-IDF"""
import os
import pathlib
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk

def is_stopwords_downloaded() -> None:
    """Are stopwords downloaded = True return already downloaded else adds stopwords"""
    try:
        nltk.data.find('corpora/stopwords')
        print("Stopwords corpus is already downloaded.")
    except LookupError:
        print("Stopwords corpus is not downloaded. Downloading now...")
        nltk.download('stopwords')
        print("Stopwords corpus has been downloaded.")
        custom_stopwords = ['also', 'day', 'make', 'one', 'ways', 'work']
        nltk_stopwords = nltk.corpus.stopwords.words('english')
        nltk_stopwords.extend(custom_stopwords)
        print("Additional values have been added after downloading the stopwords corpus.")

def list_txt_files(directory: str) -> list[str]:
    """Return a list of full paths to .txt files in the specified directory."""
    all_files = os.listdir(directory)
    txt_files = [os.path.join(directory, file) for file in all_files if file.endswith('.txt')]
    return txt_files

def file_to_string(txt_files_list: list[str]) -> str:
    """Read all content from a list of text files and concatenate into a single string."""
    all_content = ""
    # file_list= []
    for file in txt_files_list:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            all_content += content
    return all_content

def calculate_tf_idf_simplified(string_list: list[str], count_dict: dict) -> pd.DataFrame:
    """Calculate TF-IDF scores from provided document string and return DataFrame."""
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
    """Gets count of words in a string as a dict, stripping out punctuation."""
    words = re.findall(r'\b\w+\b', content.lower())
    word_counts = dict(Counter(words))
    return word_counts


def write_to_csv(output: str, fixeddf: pd.DataFrame):
    """Writes the DataFrame to a CSV"""
    output_file_path = output
    fixeddf.to_csv(output_file_path)
    return output_file_path

def create_sorted_csv_files(tf_idf_df: pd.DataFrame, output_directory: str):
    """Sort by colum write to csv"""
    # Step 1: Iterate over each column, sort it, and write top 100 rows to a new CSV
    for column in tf_idf_df.columns:
        sorted_df = tf_idf_df.sort_values(by=column, ascending=False)
        top_100_sorted_df = sorted_df.head(100)
        output_file_path = os.path.join(output_directory, f"{column}_top_100.csv")
        write_to_csv(output_file_path, top_100_sorted_df)
