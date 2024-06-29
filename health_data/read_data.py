import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    files = os.listdir("./health_data_txts")
    file_texts = []

    for file in files:
        with open(f"./health_data_txts/{file}") as f:
            file_text = f.read()
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512, chunk_overlap=64
        )
        text = text_splitter.split_text(file_text)
        file_texts.append(Document(page_content=text))

    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(
        file_texts,
        embedding=embeddings
    )
    retriever = vector_store.as_retriever()


main()


