import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS




url = 'https://www.who.int/news-room/fact-sheets/detail/mental-disorders'

response = requests.get(url)


if response.status_code == 200:
    # Open the file in write mode ('w'), which will create the file if it doesn't exist
    with open('html.txt', 'w') as f:
        # Write the content of the response to the file
        f.write(response.text)
    print("Content written to html.txt successfully.")
else:
    print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")


with open('html.txt', 'r') as f:
    content = f.read()

soup = BeautifulSoup(content, 'html.parser')

# Step 3: Extract headers and associated paragraphs
# This example assumes headers are in h2 tags and paragraphs in p tags
headers = soup.find_all(['h2', 'h3', 'h4'])  # You can add more header tags if needed

with open('parsed_content.txt', 'w') as output_file:
    for header in headers:
        output_file.write(header.get_text() + '\n')
        next_sibling = header.find_next_sibling()
        while next_sibling and next_sibling.name == 'p':
            output_file.write(next_sibling.get_text() + '\n')
            next_sibling = next_sibling.find_next_sibling()
        output_file.write('\n')

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(
    parsed_content,
    embedding=embeddings
)
