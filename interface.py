import gradio as gr
from rag import rag


asistente = rag()

interface = gr.Chatbot(label="Chat time!", type="tuples")
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(type="tuples", fn=asistente.adapter, chatbot=interface, title="RAG chatbot de prueba", examples=["Hola", "Dime algo sobre Franz Kafka", "Como se debería manejar una clase de 30 alumnos de primaria."])
    

if __name__ == "__main__":
    demo.launch() 