import chromadb

chroma_client = chromadb.PersistentClient(path='./chroma/chroma.db')
collection = chroma_client.get_or_create_collection(name='sales_messages')

# Create a document
doc = collection.create_document(
    content='Simply American author relate your.',
    metadata={'id': 1, 'title': 'Sales Message'}
)

# Query for documents
docs = collection.query('Simply American author relate your.')