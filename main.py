""" Runs full code """
import os
from dotenv import load_dotenv
from IndependentStudy import list_txt_files, file_to_string, calculate_tf_idf_simplified

def __main__():
    load_dotenv()
    input_directory = os.getenv('INPUT_DIRECTORY')
    txt_files_list = list_txt_files(input_directory)
    document_string = file_to_string(txt_files_list) # Read all content from files and concatenate into a single string
    fixeddf = calculate_tf_idf_simplified(document_string) # Pass the single string to the function
    print(fixeddf)

__main__()
