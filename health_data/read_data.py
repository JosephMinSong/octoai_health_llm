from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document


def main():
    file_text = []
    header = False
    try:
        f = open("parsed_content.txt")
    except OSError as e:
        print(e)
        return

    for line in f:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1024, chunk_overlap=64
        )
        text = text_splitter.split_text(line)
        print(text)


main()
