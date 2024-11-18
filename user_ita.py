import streamlit as st
import os
import falcon_ita as falcon
import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from dotenv import load_dotenv
load_dotenv()
#token = os.getenv('API_KEY')
#progress_bar
#main_placeholder = st.empty()
#def main_place(message="The Task Is Finished !!!!"):
  # main_placeholder.text(message)
  

st.set_page_config(page_title="TensioCare", page_icon="assets/icona.ico")


token=os.getenv("API_KEY")
# Memory management functions
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

def main():
    # Call memory management functions before starting Streamlit app
    #min_memory_available = 1 * 1024 * 1024 * 1024  # 1GB
    clear_gpu_memory()
    #wait_until_enough_gpu_memory(min_memory_available)

    st.sidebar.title("Seleziona: ")
    selection = st.sidebar.radio("Vai a: ", ["Chi Siamo","Chatta con TensioCare"])

    if selection == "Chi Siamo":
        display_chi_siamo_page()

    elif selection == "Chatta con TensioCare":
        display_chatbot_page()
   

def display_chatbot_page():
    
    st.title("TensioCare")  # Aggiungi il testo "TensioCare" accanto all'immagine
          

    existing_vector_store = "TensioCare_VS"
    temperature = 0.3
    max_length = 300

    # Prepare the LLM model
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if token:
        st.session_state.conversation = falcon.prepare_rag_llm(
            token, existing_vector_store, temperature, max_length
        )

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Source documents
    if "source" not in st.session_state:
        st.session_state.source = []

    # Mostra il messaggio iniziale solo se non è ancora stato mostrato
    if not st.session_state.get('message_shown', False):
        st.session_state.message_shown = True
        with st.chat_message(name = "TensioCare", avatar = "assets/TensioCare2.jpg"):
            st.markdown("In cosa posso esserti utile?")
    
    
    # Display chats
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ask a question
    if question := st.chat_input("Ask a question"):
        # Append user question to history
        st.session_state.history.append({"role": "user", "content": question})
        # Add user question
        with st.chat_message(name = "user", avatar = "assets/user.png"):
            st.markdown(question)

        # Answer the question
        answer, doc_source = falcon.generate_answer(question, token)
       # st.image()
        with st.chat_message(name = "TensioCare", avatar = "assets/TensioCare2.jpg"):
            st.write(answer)
        # Append assistant answer to history
        st.session_state.history.append({"role": "TensioCare", "content": answer})


def display_chi_siamo_page():
    # Crea due colonne affiancate
    col1, col2 = st.columns([1, 3])  # La prima colonna sarà più piccola, la seconda più grande

    # Colonna 1: Aggiungi l'immagine
    with col1:
        st.image("assets/TensioCare.jpeg", width=200, use_container_width=False)
        

    # Colonna 2: Aggiungi il titolo
    with col2:
        st.title("TensioCare, il tuo assistente digitale per la pressione arteriosa!")

    # Sottotitolo
    st.header("Chi siamo")

    # Descrizione introduttiva
    st.markdown("""
    Benvenuto in TensioCare, il nostro chatbot dedicato alla salute cardiovascolare!
    Il nostro obiettivo è fornire **informazioni accurate**, **facili da comprendere** e **sempre aggiornate** sull'ipertensione arteriosa, un problema di salute che colpisce milioni di persone in tutto il mondo.
    Se ti stai chiedendo cos'è l'ipertensione, come riconoscerla, quali sono i rischi associati o come gestirla, sei nel posto giusto!
    """)

    # Sezione sulle funzionalità del chatbot
    st.header("Cosa fa il nostro chatbot")

    st.markdown("""
    TensioCare è progettato per rispondere alle tue domande sull'ipertensione in modo rapido e comprensibile. Può aiutarti a:
    - **Comprendere cos'è l'ipertensione arteriosa**: definizione, cause e sintomi.
    - **Conoscere i fattori di rischio**: stile di vita, alimentazione, genetica e altre condizioni mediche.
    - **Capire come misurare la pressione arteriosa**: imparare a leggere i risultati e capire cosa significano.
    - **Scoprire le opzioni di trattamento**: farmaci, cambiamenti nello stile di vita e altre strategie per il controllo della pressione.
    - **Raccogliere informazioni preventive**: suggerimenti per ridurre il rischio di sviluppare l'ipertensione.
    """)

    # Sezione su come funziona
    st.header("Come funziona")

    st.markdown("""
    TensioCare utilizza un linguaggio **semplice** e **accessibile** per spiegare concetti medici complessi in modo che tutti possano comprenderli. È alimentato da una base di conoscenze supportata da **fonti scientifiche** e **linee guida ufficiali**.

    Se hai domande specifiche, il chatbot è sempre pronto a darti risposte affidabili e chiare.
    """)

    # Sezione sulla salute e la consultazione medica
    st.header("La tua salute, la nostra priorità")
    col1, col2 = st.columns([2,1])  # La prima colonna sarà più piccola, la seconda più grande

    # Colonna 1: Aggiungi l'immagine
    with col1:
        st.markdown("""
            Siamo consapevoli dell'importanza di affrontare l'ipertensione con consapevolezza e prevenzione.
            Il nostro chatbot non sostituisce il parere di un medico, ma ti fornisce informazioni preziose che ti aiuteranno a fare scelte informate per il tuo benessere.

            **Se hai dubbi o necessiti di una consulenza personalizzata**, ti invitiamo a rivolgerti al tuo medico di fiducia.
            """)
        
        

    # Colonna 2: Aggiungi il titolo
    with col2:
        st.image("assets/image1.jpeg")

    


if __name__ == "__main__":
    main()
