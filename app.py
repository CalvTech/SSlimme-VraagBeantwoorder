import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

st.set_page_config(page_title="🧠 Slimme VraagBeantwoorder (ENG)", page_icon="❓", layout="centered")

st.title("🧠 Slimme VraagBeantwoorder (Engels)")
st.write("Voer een Engelse tekst en vraag in. Ontvang direct het antwoord!")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

qa_pipeline = load_model()

context = st.text_area("📄 Paste your English context here:", height=250)
question = st.text_input("❓ Ask a question:")

if st.button("🔍 Find Answer"):
    if context.strip() and question.strip():
        with st.spinner("🧠 Searching for answer..."):
            answer = qa_pipeline({
                'context': context,
                'question': question
            })

        st.success("✅ Answer:")
        st.write(answer['answer'])
    else:
        st.error("⚠️ Please provide both context and question.")
