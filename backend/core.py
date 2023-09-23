import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"]
)

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str) -> any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=chat,
                                     chain_type="stuff",
                                     retriever=docsearch.as_retriever(),
                                     return_source_documents=True)
    return qa({"query": query})


if __name__ == "__main__":
    result = run_llm(query="what is langchain chain")
    print(result)
    print(result["result"])
    print(result["source_documents"])
