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



def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)

def LLMtoList(text: str) -> List[str]:
    lines = text.strip().split("\n")
    return list(filter(None, lines))  # Remove empty lines


llm = ChatOllama(base_url="http://localhost:11434", model="llama3.1:8b-instruct-q4_K_M")
multyQueryGenrator = ChatOllama(base_url="http://localhost:11434", model="llama3.2:1b")


cliente = chromadb.HttpClient("localhost", 7888)

print("conexión")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma = Chroma(collection_name="multi-qa-mpnet-base-dot-v1", embedding_function=embeddings, client=cliente)
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

query_gen_prompt = {
    "Eres un asistente de modelo de lenguaje de inteligencia artificial. Tu tarea es generar cinco versiones diferentes de la pregunta del usuario para recuperar documentos relevantes de una base de datos vectorial. Al generar múltiples perspectivas sobre la pregunta del usuario, tu objetivo es ayudarle a superar algunas de las limitaciones de la búsqueda de similitud basada en distancia. Proporciona estas preguntas alternativas separadas por saltos de línea."
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{history}"),
        ("human", "{input}"),
    ]
)

def retrieve(state: State):
    retrieved_docs = []
    raw_queries = multyQueryGenrator.invoke(state["question"])
    queries = LLMtoList(raw_queries)
    for query in queries:
        retrieved_docs.extend(chroma.similarity_search(query))
    print("Documentos encontrados: \n" + str(retrieved_docs) + "\n\n")
    return {"context": retrieved_docs}

def generate(state: State):  
    docs_content = format_docs(state["context"])
    messages = prompt.ainvoke({"input": state["question"], "context": docs_content})
    response = llm.ainvoke(messages)    
    print(("Respuesta generada"))
    return {"answer": response}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def adapter(message: str, history: list[list[str,str]]):
    result = graph.invoke({"question": message, "history": history})
    print(result["context"])
    return result["answer"].content  + "\nLos documentos originale son los siguientes:\n\n\t" + "\n\t".join([doc.metadata["source"] for doc in result   ["context"]])



interface = gr.Chatbot(label="Chat time!", type="tuples")
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(type="tuples", fn=adapter, chatbot=interface, title="RAG chatbot de prueba", examples=["Hola", "Dime algo sobre Franz Kafka", "Como se debería manejar una clase de 30 alumnos de primaria."])

if __name__ == "__main__":
    demo.launch() 