import gradio as gr
from rag import rag


def update(complexity: bool, rag: rag):
    rag.complexity = complexity

asistente = rag()

interface = gr.Chatbot(label="Chat time!", type="tuples")
with gr.Blocks() as app:
    chatbot = gr.ChatInterface(type="tuples", fn=asistente.adapter, chatbot=interface, title="RAG chatbot de prueba", examples=["Hola", "Dime algo sobre Franz Kafka", "Como se debería manejar una clase de 30 alumnos de primaria."])
    check = gr.Checkbox(value=True, show_label=True, label="Búsqueda mejorada", info="Aumenta la calida a costa de mayor tiempo de procesamiento")
    check.input(rag.get_func_update(asistente), inputs=check, trigger_mode="once")

if __name__ == "__main__":
    app.launch(server_name="Asistente_IA")

    