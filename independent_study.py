"""Generates DataFrame which includes the TF-IDF"""
import os
import pathlib
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def ensure_nltk_resources() -> None:
    """Ensure NLTK resources are downloaded."""
    try:
        nltk.data.find('corpora/stopwords')
        print("Stopwords corpus is already downloaded.")
    except LookupError:
        print("Stopwords corpus is not downloaded. Downloading now...")
        nltk.download('stopwords', force=True)
        print("Stopwords corpus has been downloaded.")
    try:
        nltk.data.find('corpora/wordnet.zip')
        print("WordNet corpus is already downloaded.")
    except LookupError:
        print("WordNet corpus is not downloaded. Downloading now...")
        nltk.download('wordnet')
        print("WordNet corpus has been downloaded.")
    try:
        nltk.data.find('tokenizers/punkt')
        print("Punkt tokenizer is already downloaded.")
    except LookupError:
        print("Punkt tokenizer is not downloaded. Downloading now...")
        nltk.download('punkt')
        print("Punkt tokenizer has been downloaded.")

def list_txt_files(directory: str) -> list[str]:
    """Return a list of full paths to .txt files in the specified directory."""
    all_files = os.listdir(directory)
    txt_files = [os.path.join(directory, file) for file in all_files if file.endswith('.txt')]
    return txt_files

def file_to_combined_string(txt_files_list: list[str]) -> str:
    """Read all content from a list of text files and concatenate into a single string."""
    all_content = ""
    for file in txt_files_list:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            all_content += content
    return all_content

def calculate_df(txt_files_list: list[str], count_dict: dict) -> pd.DataFrame:
    """Calculate TF-IDF scores from provided document string and return DataFrame with tf-idf and count of words."""
    custom_stopwords = ['et','al']
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    nltk_stopwords.extend(custom_stopwords)
    vectorizer = TfidfVectorizer(stop_words=nltk_stopwords, input='filename')
    vectors = vectorizer.fit_transform(txt_files_list)
    feature_names = vectorizer.get_feature_names_out()
    headers = [pathlib.Path(string).name for string in txt_files_list]
    words_df = pd.DataFrame(vectors.toarray(), columns=feature_names).transpose()
    words_df.columns = headers
    words_df.index.name = "Words"
    words_df['WordCounts'] = words_df.index.map(count_dict).astype(int)
    cols = words_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('WordCounts')))
    words_df = words_df[cols]
    return words_df

def count_of_words(content: str) -> dict:
    """Gets count of words in a string as a dict, stripping out punctuation."""
    words = re.findall(r'\b\w+\b', content.lower())
    word_counts = dict(Counter(words))
    return word_counts

def write_to_csv(output: str, fixeddf: pd.DataFrame):
    """Writes the DataFrame to a CSV"""
    output_file_path = output
    fixeddf.to_csv(output_file_path,index=False)
    return output_file_path

def create_sorted_csv_files(tf_idf_df: pd.DataFrame, output_directory: str):
    """Sort by colum write to csv"""
    for column in tf_idf_df.columns[1:]:
        sorted_df = tf_idf_df.sort_values(by=column, ascending=False)
        top_100_sorted_df = sorted_df.head(100)
        output_file_path = os.path.join(output_directory, f"{column}_top_100.csv")
        write_to_csv(output_file_path, top_100_sorted_df)

def pdf_to_text(input_folder, output_folder):
    """Converts PDF to TXT from folder."""
    lemmatizer = WordNetLemmatizer()
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            output_file_name = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, output_file_name)
            with open(pdf_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                text = ''
                for page in reader.pages:
                    page_text = page.extract_text() + ' '  
                    text += page_text
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                tokens = word_tokenize(text)
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
                lemmatized_text = ' '.join(lemmatized_tokens)
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(lemmatized_text)

def prep_df(df:pd.DataFrame) -> tuple[pd.DataFrame,list]:
    """preps df for graphing and grabs headders"""
    top = list(df.columns)
    count = len(top)
    for i in range(1,count):
        df[top[i]] = df[top[i]].astype(float)
    return df, top

def get_graph_vals(df:pd.DataFrame,top) -> float:
    """makes vars needed for graphing"""
    tf1 = df.sort_values(by=top[3], ascending=False)
    highest_count = df.sort_values(by=top[1], ascending=False)
    return tf1, highest_count

def barh(a,x,y,top):
    """creats bar chart"""
    width = .35
    plt.barh(a[top[x]][:10], a[top[y]][:10],width, color='gray')
    plt.title("Count of Words")
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    plt.show()

def scatter(a,x,y,top):
    """creates scatter chart"""
    s = 2
    plt.axis([0, 100, 0, .28])
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    plt.scatter(a[top[x]], a[top[y]], s =s, color = 'r')
    plt.title("Word count & TIDF of doc 1")
    plt.show()
 
def pie(a,x,y,top):
    """creates pie chart"""
    plt.pie(a[top[y]][:10], labels=a[top[x]][:10],
            autopct='%1.1f%%', pctdistance=0.84,          
            wedgeprops= {
                "edgecolor":"black",
                'linewidth': 1,
                'antialiased': True
                }
            )
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Top Word counts')
    plt.show()

def append_word_counts(df: pd.DataFrame, txt_files_list: list[str]) -> pd.DataFrame:
    """Reads each file, counts the words, and appends word counts to the existing DataFrame"""
    for file in txt_files_list:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            words = re.findall(r'\b\w+\b', content.lower())
            word_count_dict = dict(Counter(words))
            file_name = file.split('\\')[-1] 
            for word, count in word_count_dict.items():
                column_name = f"{file_name}_WordCount"
                if word in df.index:
                    if column_name in df.columns:
                        df.at[word, column_name] = count
                    else:
                        df[column_name] = 0  
                        df.at[word, column_name] = count

    return df
