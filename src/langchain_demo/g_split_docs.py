from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from config import load_config
from langchain_core.documents import Document


load_config()
data_dir = Path(__file__).parent / "data"
sample_text = data_dir / "sample_text.txt"

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        return [
            Document(page_content=text, metadata={"source": str(file_path)}),
        ]


def main(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents=docs)
    page_content_list = []
    for chunk in chunks:
        page_content_list.append(chunk.page_content)


    #embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    # embeddings = embedding.embed_documents(texts=page_content_list)
    # print(embeddings)


if __name__ == '__main__':
    file_content = load_text_file(file_path=sample_text)
    print(file_content[0].page_content)
    main(docs=file_content)