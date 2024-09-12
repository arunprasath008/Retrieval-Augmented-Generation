from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="sales_messages.csv")
documents = loader.load()

embeddings = OllamaEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
llm = Ollama(base_url="http://localhost:11434",model="samantha-mistral")

template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, ton of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect in german language:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

message = """
Hello Daniel,

I am available to catch up with you next week or via Zoom. Please let me know what works best for you.

Best,
Nantha
"""

response = generate_response(message)
print(response)

