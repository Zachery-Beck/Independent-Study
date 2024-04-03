"""Main File"""
import os
from dotenv import load_dotenv
from independent_study import list_txt_files, file_to_string, calculate_tf_idf_simplified
from independent_study import ensure_nltk_resources,count_of_words,write_to_csv,create_sorted_csv_files,pdf_to_text
def __main__():
    load_dotenv()
    input_folder = os.getenv('PDF_INPUT_FOLDER')
    output_folder = os.getenv('TXT_OUTPUT_FOLDER')
    input_directory = os.getenv('INPUT_DIRECTORY')
    output_directory = os.getenv('OUTPUT_DIRECTORY')
    ensure_nltk_resources()
    pdf_to_text(input_folder, output_folder)
    txt_files_list = list_txt_files(input_directory)
    string_list = file_to_string(txt_files_list)
    count_dict = count_of_words(string_list)
    fixeddf = calculate_tf_idf_simplified(txt_files_list,count_dict)
    output_file_path=write_to_csv('output.csv',fixeddf)
    print(f"DataFrame has been written to {output_file_path}")
    print(fixeddf)
    create_sorted_csv_files(fixeddf, output_directory)
__main__()

