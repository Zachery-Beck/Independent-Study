"""Main File"""
import os
from dotenv import load_dotenv
from IndependentStudy import list_txt_files, file_to_string, calculate_tf_idf_simplified, is_stopwords_downloaded,count_of_words,write_to_csv

def __main__():
    load_dotenv()
    is_stopwords_downloaded()
    input_directory = os.getenv('INPUT_DIRECTORY')
    txt_files_list = list_txt_files(input_directory) 
    string_list = file_to_string(txt_files_list)
    count_dict = count_of_words(string_list)
    fixeddf = calculate_tf_idf_simplified(txt_files_list,count_dict)
    output_file_path=write_to_csv('output.csv',fixeddf)
    print(f"DataFrame has been written to {output_file_path}")

__main__()

