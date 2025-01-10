import gradio as gr
import chromadb 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import chromadb
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)



llm = ChatOllama(base_url="http://localhost:11434", model="llama3.1:8b-instruct-q4_K_M")

cliente = chromadb.HttpClient("localhost", 7888)

print("conexión")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma = Chroma(collection_name="prueba_kafka", embedding_function=embeddings, client=cliente)
retriever  = chroma.as_retriever()

# from langchain_core.documents import Document
# from typing_extensions import List, TypedDict

# class State(TypedDict):
#     question: str
#     context: List[Document]
#     history: list[list[str,str]]
#     answer: str

class State(TypedDict): 
    question: str
    context: List[Document]
    answer: str
    history: str 


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
        ("placeholder", "{history}"),
        ("human", "{input}"),
    ]
)

def retrieve(state: State):
    retrieved_docs = chroma.similarity_search(state["question"])
    print("Documentos encontrados: \n" + str(retrieved_docs) + "\n\n")
    return {"context": retrieved_docs}

def generate(state: State): 
    docs_content = format_docs(state["context"])
    messages = prompt.invoke({"input": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    print(("Respuesta generada"))
    return {"answer": response}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def adapter(message: str, history: list[list[str,str]]):
    result = graph.invoke({"question": message, "history": history})
    print(result["context"])
    return result["answer"].content  + "\nLos documentos originale son los siguientes:\n\n\t" + "\n\t".join([doc.metadata["source"] for doc in result   ["context"]])

from IPython.display import Image, display

Image(graph.get_graph().draw_mermaid_png())

# prompt = ChatPromptTemplate([
#         ( """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#          Question: {question} 
#          Context: {context} 
#          Answer:""")
# ])


interface = gr.Chatbot(label="Chat time!", type="tuples")
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(type="tuples", fn=adapter, chatbot=interface, title="RAG chatbot de prueba", examples=["Hola", "Dime algo sobre Franz Kafka", "Como se debería manejar una clase de 30 alumnos de primaria."])

if __name__ == "__main__":
    demo.launch() 