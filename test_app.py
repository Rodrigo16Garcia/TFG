import gradio as gr
import chromadb 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


llm = Ollama(base_url="http://localhost:11434", model="llama3.1:8b-instruct-q4_K_M")
llm

cliente = chromadb.HttpClient("localhost", 7888)
chroma = Chroma(collection_name="prueba_kafka", embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), client=cliente)
retriever  = chroma.as_retriever()


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# prompt = ChatPromptTemplate([
#         ( """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#          Question: {question} 
#          Context: {context} 
#          Answer:""")
# ])

from langchain.globals import set_verbose
set_verbose(True)
chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)   



def predict(message: str, history: list[list[str,str]]):

    # print(prompt.({"context": "contexto","input": message}))

    print(history)
    
    return str(chain.invoke(message))


interface = gr.Chatbot(label="Chat time!")
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(fn=predict, chatbot=interface, title="RAG chatbot de prueba")

if __name__ == "__main__":
    demo.launch() 