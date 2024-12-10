import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

path = './data'
output_path = os.path.join(path, "all_text_data.txt")
text_path = os.path.join(path, "only_text_data.txt")

files =  [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
all_text_data = []


def text_data(text_path):
    """
    텍스트 파일 로드 및 청크 분할
    """
    loader = TextLoader(text_path, encoding='utf-8')
    data = loader.load()

    text_split = CharacterTextSplitter(
        separator='',
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    texts = text_split.split_text(data[0].page_content)
    return texts

def pdf_data(pdf_path):
    """
    PDF 파일 로드 및 청크 분할
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    for d in documents:
        d.metadata['file_path'] = pdf_path

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)

    # Document 객체의 page_content만 추출
    return [doc.page_content for doc in chunked_documents]

# 모든 파일 처리
for file in files:
    file_path = os.path.join(path, file)
    if file.endswith('.txt'):
        all_text_data.extend(text_data(file_path))
    elif file.endswith('.pdf'):
        all_text_data.extend(pdf_data(file_path))
    else:
        print(f"E: It isn't text or pdf: {file}")

# all_text_data를 텍스트 파일로 저장
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines("\n".join(all_text_data))

print(f"All text data saved to {output_path}.")

only_text = text_data('./data/data.txt')

with open(text_path, "w", encoding="utf-8") as f:
     f.writelines("\n".join(only_text))
    