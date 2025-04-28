import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Pagina instellingen
st.set_page_config(page_title="🧠 Slimme VraagBeantwoorder (NL)", page_icon="❓", layout="centered")

st.title("🧠 Slimme VraagBeantwoorder (Nederlands)")
st.write("Voer een Nederlandse tekst en een vraag in. Ontvang direct het antwoord! (Maximaal ongeveer **300 woorden** tekst).")

# Model laden
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("wietsedv/bert-base-dutch-cased-finetuned-squad")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

qa_pipeline = load_model()

# Tekst en vraag invoer
context = st.text_area("📄 Plak hier je Nederlandse tekst (context):", height=250)
question = st.text_input("❓ Stel je vraag over de tekst:")

# Woordenteller context
word_count = len(context.split())
st.write(f"✏️ Aantal woorden in tekst: **{word_count}** (advies: max 300)")

# Submit button
if st.button("🔍 Zoek antwoord"):
    if context.strip() and question.strip():
        if word_count > 300:
            st.warning(f"⚠️ Je tekst bevat {word_count} woorden. Probeer onder de 300 woorden te blijven voor de beste resultaten!")

        with st.spinner("🧠 Antwoord wordt gezocht..."):
            answer = qa_pipeline({
                'context': context,
                'question': question
            })

        st.success("✅ Antwoord:")
        st.write(answer['answer'])
    else:
        st.error("⚠️ Vul zowel een tekst als een vraag in!")
