import chromadb 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama 
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from typing import Literal
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
import pickle




def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)

def LLMtoList(text: str) -> List[str]:
    lines = text.strip().split("\n")
    return list(filter(None, lines))  # Remove empty lines          

class State(TypedDict): 
    question: str
    queries: List[str]
    context: List[Document]     
    answer: str
    history: List[List[str]] 
    complex: bool = True


class rag():
    
    def __init__(self):
        self.complexity = True

        self.llm = ChatOllama(base_url="http://localhost:11434", model="llama3.1:8b-instruct-q4_K_M")
        self.multyQueryGenrator = ChatOllama(base_url="http://localhost:11434", model="llama3.2:1b")


        cliente = chromadb.HttpClient("localhost", 7888)

        print("conexión")

        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        chroma = Chroma(collection_name="prueba_kafka_child_augmented", embedding_function=embeddings, client=cliente, collection_metadata={"hnsw:space": "l2", "hnsw:search_ef": 20})
        store = pickle.load(open("parent_store.pkl", 'rb'))

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size= 2000, chunk_overlap=200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size= 400)

        self.retriever = ParentDocumentRetriever(
            vectorstore=chroma,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        system_prompt = (
        "Eres un asistente cuya tarea es resolver las preguntas del usuario. "
        "Usa la información de contexto que se indica a continuación para realizar tu tarea"
        " y responder las preguntas. Si no conoces la respuesta o no aparece en el contexto "
        "solo di que no conoces la respuesta. "
        "Puedes explayarte lo que sea necesario hasta cinco párrafos."
        "\n\n"
        "{context}")

        self.query_gen_prompt = PromptTemplate(input_variables=["question"],
        template="Eres un asistente de modelo de lenguaje de inteligencia artificial. Tu tarea es generar cinco versiones diferentes de la pregunta del usuario para recuperar documentos relevantes de una base de datos vectorial. Al generar múltiples perspectivas sobre la pregunta del usuario, tu objetivo es ayudarle a superar algunas de las limitaciones de la búsqueda de similitud basada en distancia. Proporciona estas preguntas alternativas separadas solo por saltos de línea, sin nigún texto adicional. La salida debe seguir un formato similar al siguiente ejemplo:\n \"¿Cuál es la forma principal de garantizar que los estudiantes con discapacidad sea accesible a la educación?\n¿Qué tipo de tecnologías o recursos se utilizan para facilitar la inclusión de estudiantes con discapacidad?\n¿Cómo se implementan medidas para brindar acceso igualitario en el sistema educativo?\n¿Qué medidas específicas se han implementado en la comunidad educativa para promover la inclusión y el acceso igualitario a la educación?\n¿Cómo se han diseñado programas y políticas que beneficien a los estudiantes con discapacidad?\". Pregunta original: {question}")

        self.prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{history}"),
            ("human", "{input}"),
        ])

        graph_builder = StateGraph(State)
        for node in [self.gen_query ,self.retrieve, self.rerank, self.generate]:
            graph_builder.add_node(node)


        graph_builder.add_conditional_edges(START, self.fork1)
        graph_builder.add_edge("gen_query", "retrieve")
        graph_builder.add_conditional_edges("retrieve", self.fork2)
        graph_builder.add_edge("rerank", "generate")
        graph_builder.set_finish_point("generate")
        self.graph = graph_builder.compile()


        self.graph_builder = StateGraph(State).add_sequence([self.gen_query ,self.retrieve, self.rerank, self.generate])
        self.graph_builder.set_conditional_entry_point( (lambda state: "gen_query" if state["complex"] else "retrieve") )
        self.graph_builder.add_edge(START, "gen_query")
        self.graph_builder.add_edge("generate", END)
        self.graph = self.graph_builder.compile()
        

    def fork1(state: State) -> Literal["gen_query", "retrieve"]:
        if state['complex']:
            return "gen_query"
        else:
            return "retrieve"

    def fork2(state: State) -> Literal["rerank", "generate"]:
        if state['complex'] == True:
            return "rerank"
        else:
            return "generate"

    def gen_query(self, state: State):
        prompt = self.query_gen_prompt.invoke(state["question"])
        raw_queries = self.multyQueryGenrator.invoke(prompt)
        queries = LLMtoList(raw_queries.content)
        queries.append(state["question"])
        print(queries)
        return {"queries": queries}

    def retrieve(self, state: State):
        retrieved_docs = []
        try:
            for query in state["queries"]:
                retrieved_docs.extend(self.retriever.invoke(query)) 
        except KeyError:
                retrieved_docs.extend(self.retriever.invoke(state["question"])) 

        print("Documentos encontrados: \n" + str(retrieved_docs) + "\n\n")
        return {"context": retrieved_docs}

    def rerank(self, state: State): 
        print("Tamaño contexto inicial: ", len(state["context"]))
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        compressor = CrossEncoderReranker(model=model, top_n=4)
        final_context = compressor.compress_documents(state["context"], state["question"])
        print("Documentos finales: ", final_context)
        return {"context": final_context} 

    def generate(self, state: State):   
        print("Tamaño contexto final: ", len(state["context"]))
        docs_content = format_docs(state["context"])
        print("Contexto final: ", docs_content) 
        messages = self.prompt.invoke({"input": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)    
        print(("Respuesta generada: ", response))
        return {"answer": response}


    def adapter(self, message: str, history: list[list[str,str]]) -> str:
        result = self.graph.invoke({"question": message, "history": history, "complex": self.complexity})
        # print(result["context"])
        return result["answer"].content  + "\nLos documentos originale son los siguientes:\n\n\t" + "\n\t".join([doc.metadata["source"] for doc in result   ["context"]])


