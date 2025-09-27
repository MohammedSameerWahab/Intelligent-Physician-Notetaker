from transformers import pipeline
from typing import Dict
import re # Used for sentence splitting

class SentimentAnalyzer:
    """
    Analyzes the patient's final sentiment and intent at the conclusion of a
    medical conversation using a zero-shot classification model.
    This version is robust and works on speaker-agnostic text from transcripts.
    """

    def __init__(self, model_name="facebook/bart-large-mnli"):
        """
        Initializes the zero-shot classification pipeline.
        """
        print("Initializing Zero-Shot Classification pipeline...")
        try:
            # Note: Using st.cache_resource would be ideal here if called within Streamlit,
            # but we keep it independent for modularity. The app's structure
            # only initializes this class once per click, which is efficient enough.
            self.classifier = pipeline("zero-shot-classification", model=model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.sentiment_labels = ["anxious", "neutral", "reassured", "concerned"]
        self.intent_labels = [
            "reporting symptoms",
            "seeking reassurance",
            "providing information",
            "expressing gratitude",
            "asking a question"
        ]

    def analyze_final_state(self, transcript_text: str) -> Dict:
        """
        Analyzes the final sentiment and intent of the patient based on the
        concluding sentences of the conversation.

        Args:
            transcript_text (str): The raw text of the transcript.

        Returns:
            A dictionary with the overall final sentiment and intent.
        """
        if not transcript_text:
            return {"error": "Transcript text is empty."}

        # --- LOGIC FIX: Analyze last few sentences instead of searching for "Patient:" ---
        # Split the text into sentences using a simple regex.
        sentences = re.split(r'(?<=[.?!])\s+', transcript_text)
        
        # Take the last 3 sentences to capture the end of the conversation.
        # Use min() to handle transcripts with fewer than 3 sentences.
        num_sentences_to_analyze = min(len(sentences), 7)
        final_context = " ".join(sentences[-num_sentences_to_analyze:])

        if not final_context:
            return {"message": "Not enough text to analyze."}

        print(f"Analyzing final patient context: '{final_context}'")

        # Classify the combined final context for sentiment and intent
        sentiment_result = self.classifier(final_context, self.sentiment_labels, multi_label=False)
        intent_result = self.classifier(final_context, self.intent_labels, multi_label=False)

        final_analysis = {
            "final_sentiment": sentiment_result['labels'][0],
            "final_intent": intent_result['labels'][0]
        }

        return final_analysis
