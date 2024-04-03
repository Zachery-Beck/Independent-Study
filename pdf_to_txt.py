import os
from PyPDF2 import PdfReader

def pdf_to_text(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over PDF files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            output_file_name = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, output_file_name)
            with open(pdf_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)

# Example usage
input_folder = 'pdf_input_folder'
output_folder = 'text_output_folder'

pdf_to_text(input_folder, output_folder)
