"""Main File"""
import os
from dotenv import load_dotenv
from IndependentStudy import list_txt_files, file_to_string, calculate_tf_idf_simplified, is_stopwords_downloaded

def __main__():
    load_dotenv()
    is_stopwords_downloaded()
    input_directory = os.getenv('INPUT_DIRECTORY')
    txt_files_list = list_txt_files(input_directory)
    # string_list = file_to_string(txt_files_list) 
    fixeddf = calculate_tf_idf_simplified(txt_files_list)
    output_file_path = 'output.csv'
    fixeddf.to_csv(output_file_path)
    
    print(f"DataFrame has been written to {output_file_path}")

__main__()

