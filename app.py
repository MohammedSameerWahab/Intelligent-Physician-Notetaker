import streamlit as st
import json
from src.NER import process_medical_conversation
from src.sentiment_analyzer import SentimentAnalyzer
from src.gemini_llm import get_gemini_summary
from src.soap_generator import generate_soap_note

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Physician Notetaker",
    page_icon="🩺",
    layout="wide"
)

# --- App Title and Description ---
st.title("🩺 Intelligent Physician Notetaker")
st.markdown("Upload a physician-patient conversation transcript to generate structured summaries, sentiment analysis, and a S.O.A.P. note.")

# --- Session State Initialization ---
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'advanced_summary' not in st.session_state:
    st.session_state.advanced_summary = None
if 'soap_note' not in st.session_state:
    st.session_state.soap_note = None

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a transcript file (.txt)",
    type=['txt'],
    help="Upload a plain text file containing the conversation transcript."
)

# --- Main Logic ---
if uploaded_file is not None:
    transcript_text = uploaded_file.getvalue().decode("utf-8")

    # --- Action Buttons ---
    st.subheader("Choose an Analysis to Perform")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Generating local summary..."):
                summary_result = process_medical_conversation(transcript_text)
                st.session_state.summary = summary_result

    with col2:
        if st.button("Advanced Summary with Gemini", use_container_width=True):
            with st.spinner("Generating advanced summary with Gemini..."):
                advanced_result = get_gemini_summary(transcript_text)
                st.session_state.advanced_summary = advanced_result

    with col3:
        if st.button("Sentiment Analysis", use_container_width=True):
            with st.spinner("Analyzing patient sentiment..."):
                analyzer = SentimentAnalyzer()
                sentiment_result = analyzer.analyze_final_state(transcript_text)
                st.session_state.sentiment = sentiment_result



    with col4:
        if st.button("Generate S.O.A.P. Note", use_container_width=True):
            with st.spinner("Generating S.O.A.P. note with Gemini..."):
                soap_result = generate_soap_note(transcript_text)
                st.session_state.soap_note = soap_result

    # --- Displaying Results ---
    st.markdown("---")
    st.header("Results")

    if st.session_state.summary:
        st.subheader("📄 Structured Medical Summary")
        st.json(st.session_state.summary)
    
    if st.session_state.advanced_summary:
        st.subheader("✨ Advanced Gemini Summary")
        st.json(st.session_state.advanced_summary)

    if st.session_state.sentiment:
        st.subheader("😊 Final Patient Sentiment & Intent")
        st.json(st.session_state.sentiment)

    if st.session_state.soap_note:
        st.subheader("📋 S.O.A.P. Note")
        st.json(st.session_state.soap_note)

