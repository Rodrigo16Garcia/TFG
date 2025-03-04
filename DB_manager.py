# %%
import chromadb
import uuid
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents.base import Document
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
import pickle
from time import time
import os
from sys import exit 
from re import search
from glob import glob
import traceback


   # %%

def add_docs(collection: Chroma, ampliar_db: bool):
    
    try:
        path = input("Dirección del directorio con los archivos:")
        print("La dirección es "+ path)
        if not os.path.isdir(path):
            raise Exception
        print("Dirección válida")
    except Exception as e:
        print("No es direccion válida.\n", e.__str__)
        exit("Dirección introducida no es válida.")

    try:
        loader = PyPDFDirectoryLoader(path, mode="single", silent_errors=True)
        docs = loader.load()

        path2 = (path + r"\*.txt") if not path[-1]=="\\" else (path + "*.txt")
        print("Dirección a escanear: " + path2)
        texts_names = glob(path2)
        
        for text in texts_names:
            with open(text, "r") as file:
                docs.append(Document(page_content=file.read(), metadata={"source": text}))
                print(file.read())

        print("Archivos cargados: \n\t" + "\n\t".join([x.page_content for x in docs]))
    except Exception:
        print("Asegurate de que solo haya archivos .pdf en el directorio, y de que sean seguros para analizar.\n")
        traceback.print_exc()
        exit("Dirección introducida no es válida.")

    # %%
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000)

    # %%


    with open("parent_store_final.pkl", 'rb') as inp:
        store = pickle.load(inp)

    # %%
    retriever = ParentDocumentRetriever(
        vectorstore=collection,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # %%
    print("Introduciendo archivos")
    retriever.add_documents(docs)
    print("Archivos introducidos")

    # %%
    with open("parent_store_final.pkl", 'wb') as outp:
        pickle.dump(store, outp, pickle.HIGHEST_PROTOCOL)


    # %%
    llm = ChatOllama(base_url="http://localhost:11434", model="llama3.2:1b")    
    prompt = ChatPromptTemplate.from_template("Resume el siguiente texto:\n\n{doc}")


    # %%
    if ampliar_db:
        print("Generando resúmenes")
        keys = list(store.yield_keys())
        parent_docs = store.mget(keys)
        for doc in parent_docs:
            metadata = doc.metadata
            doc = doc.page_content
            query = prompt.invoke(doc)
            resumen = llm.invoke(query)
            retriever.vectorstore.add_texts([resumen.content], [metadata])
        print("Resúmenes creados e introducidos")

    print("Los archivos han sido incorporados a la base de datos de forma existosa.")
# %%

def show_docs(collection: Chroma):
# el primer campo obtiene todos los documentos originales sin repeticiones, útil
    # np.unique(np.array([ x["source"] for x in collection.get()["metadatas"]])), len(collection.get()["ids"]), collection.get()["data"]
    metadatas = [search(r"\\([^\\]+\.[a-zA-Z]+)", source).group(1) for source in list(np.unique(np.array([ x["source"] for x in collection.get()["metadatas"]])))]
    print("Los documentos originales son: \n  - ", "\n  - ".join(metadatas), "\nHay ", len(collection.get()["ids"]), "fragmentos en la base de datos" )


# %%

def delete_coll(client, collection_name):
    try:
        client.delete_collection(collection_name)
        print("Colección eliminada correctamente")
    except Exception:
        print("No se pudo eliminar la colección")

# %%º

# código para clonar una colección, puede ser una funcionalidad

# col = client.get_or_create_collection("prueba_kafka_child")  # create a new collection with L2 (default)
# newCol = client.get_or_create_collection("prueba_kafka_child_augmented")
# existing_count = col.count()
# batch_size = 10
# for i in range(0, existing_count, batch_size):
#     batch = col.get(include=["metadatas", "documents", "embeddings"], limit=batch_size, offset=i)
#     newCol.add(ids=batch["ids"], documents=batch["documents"], metadatas=batch["metadatas"],
#                embeddings=batch["embeddings"])

# print(newCol.count())
# print(newCol.get(offset=0, limit=10))  # get first 10 documents


def main():

    # %%
    # puerto = input("Indique el puerto de conexión con la base de datos vectorial Chroma.")
    # ip = input("Introduzca la dirección IP de la base de datos.")
    puerto = 7888
    ip = "localhost"

    try:
        client = chromadb.HttpClient(ip, int(puerto))
    except Exception:
        print("No se pudo conectar a la base de datos. Inicie el contenedor Docker de Chroma o introduzca el puerto correcto.")
    
    # %%
    colecciones = [ x.name for x in client.list_collections()]
    colecciones.append("Crear nueva colección")
    colecciones_id = []

    for idx, col in enumerate(colecciones):
        colecciones_id.append(str(idx+1)+")  "+col)
    
    # print(colecciones_id)

    mensaje = "Sobre que colección quiere trabajar: \n  " + "\n  ".join(colecciones_id) + "\n\n  -> "
    seleccion = input(mensaje)

    if int(seleccion) == len(colecciones):
        mensaje_pedir_nombre = "Introduzca el nombre de la nueva colección: -> "
        nombre_col = input(mensaje_pedir_nombre)
    else:
        nombre_col = colecciones[int(seleccion)-1]

    client.get_or_create_collection(nombre_col, metadata={"hnsw:space": "cosine", "hnsw:search_ef": 20})
    # %%
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    coleccion = Chroma(collection_name=nombre_col, client=client,  embedding_function=embedding_model, collection_metadata={"hnsw:space": "cosine", "hnsw:search_ef": 20})
    # %%
    mensaje = "Que quiere hacer con la colección: \n\t1) Añadir archivos de un directorio\n\t2) Eliminar colección\n\t3) Ver documentos en colección" + "\n\n  -> "
    seleccion = input(mensaje)

    match seleccion:
        case "1":
            ampliacion = input("¿Quiere añadir resumenes de los documentos además de los fragmentos normales?\n  ATENCIÓN: esto aumentará mucho el tiempo de procesamiento de los archivos.\n   1) No quiero resúmenes\n   2) Sí, añadir resúmenes\n\n  -> ")
            match ampliacion:
                case "1":
                    ampliacion = False
                case "2":
                    ampliacion = True
                case _:
                    ampliacion = False
            add_docs(coleccion, ampliacion)
        case "2":
            delete_coll(client, nombre_col)
        case "3":
            show_docs(coleccion)
        case _:
            print("Entrada incorrecta. Cerrando gestor de DB.")

if __name__ == "__main__":
    main()

# %%
