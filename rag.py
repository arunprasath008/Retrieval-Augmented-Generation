#Retrieval Augmented Generation (RAG)
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings()
llm = Ollama(base_url="http://localhost:8080",model="llama2")

def createVectorStoreWithCsvFile(filename: str, query: str):
    loader = CSVLoader(file_path=filename)
    
    # Create an index using the loaded documents
    index_creator = VectorstoreIndexCreator(embedding=embeddings)
    docsearch = index_creator.from_loaders([loader])

    # Create a queestion-answering chain using the index
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
    response = chain({"question": query})['result']
    return response


res = createVectorStoreWithCsvFile("sales_messages.csv", "Return the row that contains 'Simply American author relate your.' in its row")
print(res)

