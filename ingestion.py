import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

index_name = "langchain-doc-index"

dirpath = os.path.dirname(__file__)
os.chdir(dirpath)
print("Current working directory: {0}".format(os.getcwd()))

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="./langchain-docs/langchain.readthedocs.io/en/latest")
    raw_docs = loader.load()
    print(f"loaded {len(raw_docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])

    documents = text_splitter.split_documents(documents=raw_docs)
    print(f"Split into {len(documents)} chunks")

    # replace the directory with https:/ so that this can be referenced
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents=documents, embedding=embeddings, index_name=index_name)
    print("****** Added to Pinecone vectorstore vectors")



if __name__ == "__main__":
    ingest_docs()
