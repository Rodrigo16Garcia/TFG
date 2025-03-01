import gradio as gr
import chromadb 
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama 
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from time import time



def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOllama(base_url="http://localhost:11434", model="llama3.1:8b-instruct-q4_K_M")

cliente = chromadb.HttpClient("localhost", 7888)

print("conexión")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma = Chroma(collection_name="all-MiniLM-L6-v2", embedding_function=embeddings, client=cliente, collection_metadata={"hnsw:space": "l2", "hnsw:search_ef": 20})

retriever  = chroma.as_retriever()


class State(TypedDict): 
    question: str
    context: List[Document]     
    answer: str
    history: List[List[str]] 


system_prompt = (
    "Eres un asistente cuya tarea es resolver las preguntas del usuario. "
    "Usa la información de contexto que se indica a continuación para realizar tu tarea"
    " y responder las preguntas. Si no conoces la respuesta o no aparece en el contexto "
    "solo di que no conoces la respuesta. "
    "Puedes explayarte lo que sea necesario hasta cinco párrafos."
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



def invoke_coll(coll_name = "all-MiniLM-L6-v2", model = "all-MiniLM-L6-v2", distance = "l2", llm_model = "llama3.1:8b-instruct-q4_K_M"):
    global llm
    llm = ChatOllama(base_url="http://localhost:11434", model=llm_model)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/"+model)

    global chroma 
    chroma = Chroma(collection_name=coll_name, client=cliente, collection_metadata={"hnsw:space": distance, "hnsw:search_ef": 20} , embedding_function=embeddings)
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    global graph
    graph = graph_builder.compile()


def print_chroma():
    print(chroma)

interface = gr.Chatbot(label="Chat time!", type="tuples")
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(type="tuples", fn=adapter, chatbot=interface, title="RAG chatbot de prueba", examples=["Hola", "Dime algo sobre Franz Kafka", "Como se debería manejar una clase de 30 alumnos de primaria."])

if __name__ == "__main__":
    demo.launch() 