from transformers import pipeline
import streamlit as st

class AudioTranscriber:
    """
    A class to handle audio transcription using the Whisper model.
    The model is loaded once and reused for efficiency.
    """
    def __init__(self, model_name="openai/whisper-base.en"):
        """Initializes the ASR pipeline."""
        try:
            # Using st.cache_resource to prevent reloading the model on every run
            @st.cache_resource
            def load_model():
                print("Initializing Whisper ASR pipeline (this will run only once)...")
                return pipeline(
                    "automatic-speech-recognition",
                    model=model_name
                )
            self.transcriber = load_model()
        except Exception as e:
            st.error(f"Error loading transcription model: {e}")
            self.transcriber = None

    def transcribe(self, audio_file) -> str:
        """
        Transcribes an audio file to text.

        Args:
            audio_file: An uploaded audio file object from Streamlit.

        Returns:
            The transcribed text as a string.
        """
        if not self.transcriber:
            return "Error: Transcription model not loaded."
            
        try:
            # Read the audio file's content in bytes
            audio_bytes = audio_file.read()
            
            # --- THE FIX IS HERE ---
            # Pass return_timestamps=True to enable long-form audio processing.
            result = self.transcriber(audio_bytes, return_timestamps=True)
            
            return result["text"].strip()
            
        except Exception as e:
            st.error(f"An error occurred during transcription: {e}")
            return f"Transcription failed. Error: {e}"

# We'll modify the app to use the class, so a separate function is no longer needed.
