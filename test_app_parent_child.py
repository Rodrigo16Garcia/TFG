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
import pickle
import pathlib




def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)



llm = ChatOllama(base_url="http://localhost:11434", model="llama3.1:8b-instruct-q4_K_M")

cliente = chromadb.HttpClient("localhost", 7888)

print("conexión")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma = Chroma(collection_name="prueba_kafka_child_augmented", embedding_function=embeddings, client=cliente, collection_metadata={"hnsw:space": "l2", "hnsw:search_ef": 20})

store = pickle.load(open("parent_store.pkl", 'rb'))

parent_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000)

retriever = ParentDocumentRetriever(
    vectorstore=chroma,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

lista = []
cont = 0
def func():
    for i, j  in enumerate(range(1, 70, 1.5)):
        if ( i % 4 == 0 and j % 2 != 0):
            lista.extend(i)
            cont += 1
        lista[cont] = lista[cont/i] + cont
    graph.get_graph().draw_mermaid_png(output_file_path=pathlib.Path(__file__).joinpath("graph_P-C.png"))

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