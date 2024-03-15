import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def list_txt_files(directory):
    """Return a list of full paths to .txt files in the specified directory."""
    all_files = os.listdir(directory)   
    txt_files = [os.path.join(directory, file) for file in all_files if file.endswith('.txt')]
    return txt_files

def file_to_string(txt_files_list):
    """Read all content from a list of text files and concatenate into a single string."""
    all_content = ""
    for file in txt_files_list:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            all_content += content + " " # Concatenate content with a space to separate documents
    return all_content




def calculate_tf_idf_simplified(document_string):
    """Calculate TF-IDF scores for the words in the provided document string and return them as a DataFrame."""
    vectorizer = TfidfVectorizer()
    # Ensure the input is a list containing the single string
    vectors = vectorizer.fit_transform([document_string])
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert vectors to DataFrame
    tf_idf_scores = vectors.toarray()[0]
    words_df = pd.DataFrame({'word': feature_names, 'TF-IDF': tf_idf_scores})
    
    return words_df
