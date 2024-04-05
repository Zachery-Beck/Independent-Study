"""Main File"""
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from independent_study import list_txt_files, file_to_string, calculate_tf_idf_simplified
from independent_study import ensure_nltk_resources,count_of_words,write_to_csv,create_sorted_csv_files
from independent_study import barh,pie,scatter,get_graph_vals,prep_df,pdf_to_text
def __main__():
    load_dotenv()
    input_folder = os.getenv('PDF_INPUT_FOLDER')
    output_folder = os.getenv('TXT_OUTPUT_FOLDER')
    input_directory = os.getenv('INPUT_DIRECTORY')
    output_directory = os.getenv('OUTPUT_DIRECTORY')
    # print(nltk.data.find('corpora/wordnet'))
    ensure_nltk_resources()
    pdf_to_text(input_folder, output_folder)
    txt_files_list = list_txt_files(input_directory)
    string_list = file_to_string(txt_files_list)
    count_dict = count_of_words(string_list)
    df = calculate_tf_idf_simplified(txt_files_list,count_dict)
    output_file_path=write_to_csv('output.csv',df)
    print(f"DataFrame has been written to {output_file_path}")
    print(df)
    create_sorted_csv_files(df, output_directory)
    df.reset_index(inplace=True)
    fixeddf,top=prep_df(df)
    tf1, highest_count=get_graph_vals(fixeddf,top)
    plt.rcParams['figure.dpi']=200
    barh(highest_count,0,1,top)
    pie(highest_count,0,1,top)
    scatter(highest_count,1,2,top)
__main__()
