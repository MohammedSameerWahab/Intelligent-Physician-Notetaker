# Import your modules
import shutil
from src.NER import process_medical_conversation
from src.sentiment_analyzer import SentimentAnalyzer
from src.gemini_llm import get_gemini_summary
from src.soap_generator import generate_soap_note
from src.transcriber import AudioTranscriber
import streamlit as st

# --- Helper function to check for FFmpeg ---
def check_ffmpeg():
    """Checks if FFmpeg is installed and in the system's PATH."""
    return shutil.which("ffmpeg") is not None

# --- NEW: Callback function to track the active uploader and reset state ---
def set_active_uploader(uploader_key):
    """
    Sets the most recently used uploader and clears old results.
    This runs BEFORE the rest of the script reruns.
    """
    st.session_state.active_uploader = uploader_key
    # Clear all previous results when a new file is uploaded
    st.session_state.transcript = ""
    st.session_state.summary = None
    st.session_state.sentiment = None
    st.session_state.advanced_summary = None
    st.session_state.soap_note = None

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Physician Notetaker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Sidebar Content ---
with st.sidebar:
    st.header("About This Project")
    st.markdown(
        """
        This application is an AI-powered tool designed to assist healthcare professionals by automating the process of documenting physician-patient conversations.
        
        **It leverages multiple NLP models to:**
        - Transcribe audio conversations using OpenAI's Whisper.
        - Extract structured medical information (NER).
        - Analyze the patient's sentiment and intent.
        - Generate advanced summaries and clinical S.O.A.P. notes using Google's Gemini.
        """
    )
    st.markdown("---")
    st.markdown("Built by a passionate developer.")
    st.markdown("Connect on [LinkedIn](https://www.linkedin.com/in/mohammed-sameer-wahab)!") # Replace with your LinkedIn profile

# --- Main App Interface ---
st.title("🩺 Intelligent Physician Notetaker")
st.markdown("#### From Conversation to Clinic-Ready Notes in Seconds")
st.markdown("---")

# --- Session State Initialization ---
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'advanced_summary' not in st.session_state:
    st.session_state.advanced_summary = None
if 'soap_note' not in st.session_state:
    st.session_state.soap_note = None
# NEW: This state variable tracks the last-used uploader.
if 'active_uploader' not in st.session_state:
    st.session_state.active_uploader = None

# --- Two-Column Upload Section ---
st.subheader("1. Upload Your Conversation File")
col1, col2 = st.columns(2)

# Column 1: Text File Upload
with col1:
    with st.container(border=True):
        st.markdown("##### 📄 Upload a Text File")
        text_file = st.file_uploader(
            "Choose a transcript file (.txt)",
            type=['txt'],
            key="text_uploader",
            # Use the new callback function
            on_change=set_active_uploader,
            args=("text_uploader",)
        )

# Column 2: Audio File Upload
with col2:
    with st.container(border=True):
        st.markdown("##### 🎙️ Upload an Audio File")
        ffmpeg_installed = check_ffmpeg()
        audio_file = st.file_uploader(
            "Choose an audio file (.wav, .mp3, .m4a)",
            type=['wav', 'mp3', 'm4a'],
            key="audio_uploader",
            disabled=not ffmpeg_installed,
            # Use the new callback function
            on_change=set_active_uploader,
            args=("audio_uploader",)
        )
        if not ffmpeg_installed:
            st.warning("Install FFmpeg to enable audio uploads.")

active_uploader_key = st.session_state.get('active_uploader')
uploaded_file = None
if active_uploader_key and st.session_state[active_uploader_key] is not None:
    uploaded_file = st.session_state[active_uploader_key]

    # Process the file only if the transcript has been cleared by the callback
    if st.session_state.transcript == "":
        if uploaded_file.type.startswith('audio/'):
            st.info("Audio file detected. Transcribing, please wait...")
            with st.spinner("Transcription in progress... This may take a minute."):
                transcriber = AudioTranscriber()
                st.session_state.transcript = transcriber.transcribe(uploaded_file)
            st.success("✅ Transcription complete!")
        else:
            st.session_state.transcript = uploaded_file.getvalue().decode("utf-8")
            st.success("✅ Text file loaded successfully!")


# --- Display Transcript and Analysis Buttons ---
if st.session_state.transcript:
    with st.expander("View Full Transcript", expanded=False):
        st.write(st.session_state.transcript)

    st.markdown("---")
    st.subheader("2. Choose an Analysis to Perform")
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    with b_col1:
        if st.button("📄 Generate Summary", use_container_width=True, type="primary"):
            with st.spinner("Generating local summary..."):
                st.session_state.summary = process_medical_conversation(st.session_state.transcript)

    with b_col2:
        if st.button("✨ Advanced Summary", use_container_width=True, type="primary"):
            with st.spinner("Generating advanced summary with Gemini..."):
                st.session_state.advanced_summary = get_gemini_summary(st.session_state.transcript)

    with b_col3:
        if st.button("😊 Sentiment Analysis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing patient sentiment..."):
                analyzer = SentimentAnalyzer()
                st.session_state.sentiment = analyzer.analyze_final_state(st.session_state.transcript)

    with b_col4:
        if st.button("📋 Generate S.O.A.P. Note", use_container_width=True, type="primary"):
            with st.spinner("Generating S.O.A.P. note with Gemini..."):
                st.session_state.soap_note = generate_soap_note(st.session_state.transcript)
    
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
        st.subheader("📋 AI-Generated S.O.A.P. Note")
        st.json(st.session_state.soap_note)

