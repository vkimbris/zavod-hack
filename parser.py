import os
from fastapi import File
from langchain_community.document_loaders import PDFMinerLoader, Docx2txtLoader
from typing import Optional

from tika import parser
import tika
tika.initVM()

def get_text_from_file(file_path: str) -> str:
    """
    Parse files (which are not .pdf or .docx)
    With tika parser
    
    Args:
    file_name (bytes): Name of the file to be uploaded.
    """
    file_data = parser.from_file(file_path)
    text = file_data['content'].strip('\n')
    return text


def parse_file_to_txt(data: bytes, filename: str) -> Optional[str]:
    """
    Recieve file bytes the save as a temp file
    Read the file via langchain reader and get string from it

    Args:
    file (bytes): The file to be uploaded.
    """
    with open(filename, 'wb') as file:
        file.write(data)
    
    text = ''
    try:
        if filename.endswith('.docx'):
            text: str = Docx2txtLoader(filename).load()[0].page_content
         
        elif filename.endswith('.pdf'):
            text: str = PDFMinerLoader(filename).load()[0].page_content
        else:
            text: str = get_text_from_file(filename)
    except:
        # failed to read text file
        return None
        
    
    try:
        os.remove(filename)
    except:
        # access denied
        pass
        
    return text