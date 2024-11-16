import streamlit as st
from langchain_community.document_loaders import TextLoader
from pypdf import PdfReader
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from translate import Translator

en_to_it = Translator(from_lang="en", to_lang="it")
it_to_en = Translator(from_lang="it", to_lang="en")


load_dotenv()
#import chatbot_streamlit_combined

def read_pdf(file):
    document = ""

    reader = PdfReader(file)
    
    for page in reader.pages:
        document += page.extract_text()

    return document, reader.metadata


def split_doc(document, metadata, chunk_size, chunk_overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    metadataL = [metadata]*len(split)


    split = splitter.create_documents(split, metadataL)

    return split


def embedding_storing( split, create_new_vs, existing_vector_store, new_vs_name):
    if create_new_vs is not None:
        # Load embeddings instructor
        #instructor_embeddings = HuggingFaceInstructEmbeddings(
           # model_name='hkunlp/instructor-xl', model_kwargs={"device":"cpu"}
       # )
        #model_name="GroNLP/gpt2-small-italian-embeddings" this is for italian language embedding
        instructor_embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Implement embeddings
        db = FAISS.from_documents(split, instructor_embeddings)

        if create_new_vs == True:
            # Save db
            db.save_local("vector store/" + new_vs_name)
        else:
            # Load existing db
            load_db = FAISS.load_local(
                "vector store/" + existing_vector_store,
                instructor_embeddings,
                allow_dangerous_deserialization=True
            )
            # Merge two DBs and save
            load_db.merge_from(db)
            load_db.save_local("vector store/" + new_vs_name)
            
        #chatbot_streamlit_combined.main_place()
        st.success("The document has been saved.")
        
        
def prepare_rag_llm(
    token, vector_store_list, temperature, max_length
):
    # Load embeddings instructor
    #instructor_embeddings = HuggingFaceInstructEmbeddings(
        #model_name='hkunlp/instructor-xl', model_kwargs={"device":"cpu"}
    #)
    #model_name="GroNLP/gpt2-small-italian-embeddings" this is for italian language embedding
    instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})
    # Load db
    loaded_db = FAISS.load_local(
        f"vector store/{vector_store_list}", instructor_embeddings, allow_dangerous_deserialization=True
    )

    # Load LLM
    #repo_id='google/flan-t5-xxl'
    #repo_id=andreabac3/Fauno-Italian-LLM-7B this is for italian language
    llm = HuggingFaceHub(
        repo_id = 'tiiuae/falcon-7b-instruct',
        model_kwargs={"temperature": temperature, "max_length": max_length},
        huggingfacehub_api_token=token
    )

    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Create the chatbot
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory,
    )

    return qa_conversation


def generate_answer(question_ita, token):
    answer_ita = "C'è stato un errore"
    # Answer the question
    question_en = it_to_en.translate(question_ita)
    
    if token == "":
        answer_ita = "Inserisci il token Hugging Face"
        doc_source = ["no source"]
    else:
        response = st.session_state.conversation({"question": question_en})
        answer_en = response.get("answer").split("Helpful Answer:")[-1].strip()
        
        explanation = response.get("source_documents", [])
        doc_source = [d.page_content for d in explanation]
        link = [m.metadata['/URL'] for m in explanation]
        
        answer_ita = en_to_it.translate(answer_en)        
        answer_ita = f"{answer_ita}\n\n**Fonti:**\n\n{link}"

    return answer_ita, doc_source
    