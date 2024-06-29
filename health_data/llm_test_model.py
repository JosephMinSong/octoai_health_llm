from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()

ector_store = FAISS.from_documents(
    parsed_content,
    embedding=embeddings
)