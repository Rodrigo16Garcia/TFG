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
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
import pickle


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)



llm = ChatOllama(base_url="http://localhost:11434", model="llama3.1:8b-instruct-q4_K_M")

cliente = chromadb.HttpClient("localhost", 7888)

print("conexión")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma = Chroma(collection_name="prueba_kafka_child", embedding_function=embeddings, client=cliente)

store = pickle.load(open("parent_store.pkl", 'rb'))

parent_splitter = RecursiveCharacterTextSplitter(chunk_size= 2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size= 400)

retriever = ParentDocumentRetriever(
    vectorstore=chroma,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

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
    print("Pregunta:", state["question"])
    retrieved_docs = retriever.invoke("Contenidos ciencias sociales sexto primaria")
    print("Documentos encontrados: \n" + str(retrieved_docs) + "\n\n")
    return {"context": retrieved_docs}

def generate(state: State): 
    docs_content = format_docs(state["context"])
    messages = prompt.invoke({"input": state["question"], "context": docs_content})
    print("Generando respuesta")
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