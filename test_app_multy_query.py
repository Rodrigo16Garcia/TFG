import gradio as gr
import chromadb 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama 
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder



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
chroma = Chroma(collection_name="all-MiniLM-L6-v2", embedding_function=embeddings, client=cliente, collection_metadata={"hnsw:space": "l2", "hnsw:search_ef": 20})
retriever  = chroma.as_retriever() 


class State(TypedDict): 
    question: str
    queries: List[str]
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

query_gen_prompt = PromptTemplate(input_variables=["question"],
    template="Eres un asistente de modelo de lenguaje de inteligencia artificial. Tu tarea es generar cinco versiones diferentes de la pregunta del usuario para recuperar documentos relevantes de una base de datos vectorial. Al generar múltiples perspectivas sobre la pregunta del usuario, tu objetivo es ayudarle a superar algunas de las limitaciones de la búsqueda de similitud basada en distancia. Proporciona estas preguntas alternativas separadas solo por saltos de línea, sin nigún texto adicional. La salida debe seguir un formato similar al siguiente ejemplo:\n \"¿Cuál es la forma principal de garantizar que los estudiantes con discapacidad sea accesible a la educación?\n¿Qué tipo de tecnologías o recursos se utilizan para facilitar la inclusión de estudiantes con discapacidad?\n¿Cómo se implementan medidas para brindar acceso igualitario en el sistema educativo?\n¿Qué medidas específicas se han implementado en la comunidad educativa para promover la inclusión y el acceso igualitario a la educación?\n¿Cómo se han diseñado programas y políticas que beneficien a los estudiantes con discapacidad?\". Pregunta original: {question}")



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{history}"),
        ("human", "{input}"),
    ]
)
    
def gen_query(state: State):
    prompt = query_gen_prompt.invoke(state["question"])
    raw_queries = multyQueryGenrator.invoke(prompt)
    queries = LLMtoList(raw_queries.content)
    print(queries)
    return {"queries": queries}

def retrieve(state: State):
    retrieved_docs = []
    for query in state["queries"]:
        retrieved_docs.extend(retriever.invoke(query)) 
    print("Documentos encontrados: \n" + str(retrieved_docs) + "\n\n")
    return {"context": retrieved_docs}

def rerank(state: State): 
    print("Tamaño contexto inicial: ", len(state["context"]))
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=model, top_n=4)
    final_context = compressor.compress_documents(state["context"], state["question"])
    print("Documentos  finales: ", final_context)
    return {"context": final_context}

def generate(state: State):   
    print("Tamaño contexto final: ", len(state["context"]))
    docs_content = format_docs(state["context"])
    print("Contexto final: ", docs_content) 
    messages = prompt.invoke({"input": state["question"], "context": docs_content})
    response = llm.invoke(messages)    
    print(("Respuesta generada: ", response))
    return {"answer": response}

graph_builder = StateGraph(State).add_sequence([gen_query ,retrieve, rerank, generate])
graph_builder.add_edge(START, "gen_query")
graph = graph_builder.compile()

def adapter(message: str, history: list[list[str,str]]):
    result = graph.invoke({"question": message, "history": history})
    # print(result["context"])
    return result["answer"].content  + "\nLos documentos originale son los siguientes:\n\n\t" + "\n\t".join([doc.metadata["source"] for doc in result   ["context"]])

interface = gr.Chatbot(label="Chat time!", type="tuples")
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(type="tuples", fn=adapter, chatbot=interface, title="RAG chatbot de prueba", examples=["Hola", "Dime algo sobre Franz Kafka", "Como se debería manejar una clase de 30 alumnos de primaria."])

demo.launch() 