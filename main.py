""" Runs full code """
import os
from dotenv import load_dotenv
from IndependentStudy import is_stopwords_downloaded, file_to_list
from IndependentStudy import tif_idf
from file_operations import list_txt_files


def __main__():
    load_dotenv()
    is_stopwords_downloaded()
    input_directory = os.getenv('INPUT_DIRECTORY')
    txt_files_list=list_txt_files(input_directory)
    all_words_list=file_to_list(txt_files_list)
    print(tif_idf(all_words_list))


__main__()