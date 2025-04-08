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



OLLAMA_URL  = "http://localhost:11434"
MAIN_LLM    = "mistral-nemo:latest"
QUERY_LLM   = "llama3.2:1b"
CHROMA_IP   = "localhost"
CHROMA_PORT = 7888
CHROMA_COLLECTION = "cos_qa-mpnet_aug"

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
    """Clase que contiene la lógica de procesamiento de las preguntas del usuario

    Métodos:
        - fork1: (state: State) -> gen_query | retrieve 
            decide si la ejecución comenzará por la generación de preguntas o si irá directo a extraer contexto de la BD
        - fork2: (state: State) -> rerank | generate
            decide si la ejecución continuará por la la reordenadción y eliminación de contextos o si irá directo a la generación de la respuesta
        - gen_query: (state: State) -> dict[str, List[str]]
            genera preguntas similares a la original y las añade al estado. Solo se usa si self.complexitiy == True
        - retrieve: (state: State) -> dict[str, List[Document]]
            recupera el contexto de la base de conocimiento a partir de la pregunta o preguntas recibidas 
        - rerank: (state: Sate) -> dict[str, List[Document]] 
            reduce todos los documentos recibidos a lo que más relevancia tengan. Solo se usa si self.complexity = True
        - generate: (state: State) -> BaseMessage
            genera la respuesta final a partir de la pregunta inicial y los contextos recuperados
        - adapter: (message: str, history: List[List[str, str]]) -> str
            adaptador de las funciones de la clase para ser compatible con Gradio
        - get_func_update: (self: rag) -> ((bool) ->  ())
            adaptador para actualizar estado del sistema desde evento de Gradio sin recurrir a funciones externas
        - update_complexity: (complexity: bool) -> ()
            actualiza la ruta a seguir durante el procesamiento de las preguntas. Solo se accede desde self.get_func_update()
    """
    
    def __init__(self):
        """Inicializa las variables de instancia y establece todas las conexiones con los distintos componentes 
        de forma automática sin necesidad de pasar argumentos
        """

        self.complexity = True

        self.llm = ChatOllama(base_url=OLLAMA_URL, model=MAIN_LLM)
        self.multyQueryGenrator = ChatOllama(base_url=OLLAMA_URL, model=QUERY_LLM)


        cliente = chromadb.HttpClient(CHROMA_IP, CHROMA_PORT)

        print("conexión")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
        chroma = Chroma(collection_name=CHROMA_COLLECTION, embedding_function=embeddings, client=cliente, collection_metadata={"hnsw:space": "cosine", "hnsw:search_ef": 20})
        store = pickle.load(open( CHROMA_COLLECTION + ".pkl", 'rb'))

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000)

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
        

    def fork1(self, state: State) -> Literal["gen_query", "retrieve"]:
        """Decide si el procesamiento de la respuesta comienza con la generación de preguntas 
        o si salta a la recuperación de contextos

        Returns:
            str: el siguiente método a ajecutar en el grafo
        """

        if state['complex']:
            return "gen_query"
        else:
            return "retrieve"

    def fork2(self, state: State) -> Literal["rerank", "generate"]:
        """Decide si el procesamiento de la respuesta continúa con la reordenación de los contextos  
        o si salta a la generación de la respuesta con menos contextos

        Returns:
            str: el siguiente método a ajecutar en el grafo
        """

        if state['complex'] == True:
            return "rerank"
        else:
            return "generate"

    def gen_query(self, state: State):
        """Método que genera cinco versiones alternativas de la pregunta original y las guarda en el estado

        Args:
            state (State): el estado completo del grafo

        Returns:
            dict[str, List[str]]: lista de preguntas generadas y la pregunta original
        """

        prompt = self.query_gen_prompt.invoke(state["question"])
        raw_queries = self.multyQueryGenrator.invoke(prompt)
        queries = LLMtoList(raw_queries.content)
        queries.append(state["question"])
        print(queries)
        return {"queries": queries}

    def retrieve(self, state: State):
        """Método que recupera los contextos apropiados de la base de conocimiento. 
        Si hay varias preguntas, usa todas y recupera 4 documentos por cada una

        Args:
            state (State): estado completo del grafo

        Returns:
            dict[str, List[Document]]: lista de documentos recuperados que forman el contexto
        """

        retrieved_docs = []
        try:
            for query in state["queries"]:
                retrieved_docs.extend(self.retriever.invoke(query)) 
        except KeyError:
                retrieved_docs.extend(self.retriever.invoke(state["question"])) 

        print("Documentos encontrados: \n" + str(retrieved_docs) + "\n\n")
        return {"context": retrieved_docs}

    def rerank(self, state: State): 
        """Método que reordena los documentos y elimina los peores clasificados según relevancia. 
        Solo se aplica si se ha hecho generación de preguntas.

        Args:
            state (State): estado completo del grafo

        Returns:
            dict[str, List[Document]]: lista de documentos filtrada final
        """

        print("Tamaño contexto inicial: ", len(state["context"]))
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        compressor = CrossEncoderReranker(model=model, top_n=4)
        final_context = compressor.compress_documents(state["context"], state["question"])
        print("Documentos finales: ", final_context)
        return {"context": final_context} 

    def generate(self, state: State):   
        """Método que genera la respuesta final a partir de la pregunta y del contexto recuperado

        Args:
            state (State): estado completo del grafo

        Returns:
            dict[str, BaseMessage]: respuesta final de la LLM, además de la pregunta inicial y los contextos usados como metadatos
        """

        print("Tamaño contexto final: ", len(state["context"]))
        docs_content = format_docs(state["context"])
        print("Contexto final: ", docs_content) 
        messages = self.prompt.invoke({"input": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)    
        print(("Respuesta generada: ", response))
        return {"answer": response}


    def adapter(self, message: str, history: list[list[str,str]]) -> str:
        """Método adaptador que permite usar las funciones de la clase sin conocer su funcionamiento.
        Fue diseñado para adaptarse a Gradio y convierte la salida final a una cadena normal

        Args:
            message (str): pregunta inicial del usuario
            history (list[list[str,str]]): lista de interacciones entre usuario y asistente represeentados como listas con un par de str cada una. 
            Primero usuario, luego asistente.

        Returns:
            str: respuesta final del asistente a la pregunta del usuario
        """

        result = self.graph.invoke({"question": message, "history": history, "complex": self.complexity})
        # print(result["context"])
        return result["answer"].content  + "\nLos documentos originale son los siguientes:\n\n\t" + "\n\t".join([doc.metadata["source"] for doc in result["context"]])

    def get_func_update(self):
        """Clase adaptador para permitir a Gradio actualizar la ruta de procesamiento a usar sin métodos externos

        Returns: 
            update_complexity(complexity: bool): función que actualiza la ruta a seguir para procesar cada pregunta
        """

        def update_complexity(complexity: bool):
            print(self, complexity, self.complexity)
            self.complexity = complexity 

    
        
        return update_complexity

