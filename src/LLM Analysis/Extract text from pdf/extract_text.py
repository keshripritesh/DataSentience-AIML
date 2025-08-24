import PyPDF2

def extract_text_from_pdf(pdf_path, output_txt_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            # Creating a PDF reader object
            reader = PyPDF2.PdfReader(pdf_file)
            
            all_text = ""
            
            # Loop through all pages
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
            
            # Save the extracted text to a txt file
            with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(all_text)
            
            print(f"Text successfully extracted to: {output_txt_path}")
    
    except FileNotFoundError:
        print("Error: PDF file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    input_pdf = "ELectrical_Circuits.pdf"   # Replace with your PDF file path
    output_txt = "output.txt"               # Replace with your desired output text file path
    extract_text_from_pdf(input_pdf, output_txt)
    
