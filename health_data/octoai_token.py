from dotenv import load_dotenv
import os
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS




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



# Load environment variables from .env file
load_dotenv()

# Access the environment variable
OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

# Use the environment variable
print(f"OCTOAI_API_TOKEN: {OCTOAI_API_TOKEN}")

llm = OctoAIEndpoint(
        model="meta-llama-3-8b-instruct",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
    )


template = """You are a mental health generalist. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)