# src/sentiment_analyzer.py

from transformers import pipeline
from typing import Dict

class SentimentAnalyzer:
    """
    Analyzes the patient's final sentiment and intent at the conclusion of a
    medical conversation using a zero-shot classification model.
    """

    def __init__(self, model_name="facebook/bart-large-mnli"):
        """
        Initializes the zero-shot classification pipeline.
        """
        print("Initializing Zero-Shot Classification pipeline...")
        try:
            self.classifier = pipeline("zero-shot-classification", model=model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Define the custom labels for our two classification tasks
        self.sentiment_labels = ["anxious", "neutral", "reassured", "concerned"]
        self.intent_labels = [
            "reporting symptoms",
            "seeking reassurance",
            "providing information",
            "expressing gratitude",
            "asking a question"
        ]

    def analyze_final_state(self, transcript_path: str) -> Dict:
        """
        Analyzes the final sentiment and intent of the patient based on their
        concluding remarks in the conversation.

        Args:
            transcript_path (str): The full path to the transcript file.

        Returns:
            A dictionary with the overall final sentiment and intent.
        """
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                full_transcript = f.read()
        except FileNotFoundError:
            print(f"Error: Transcript file not found at {transcript_path}")
            return {"error": f"File not found at {transcript_path}"}

        # 1. Isolate all patient utterances
        patient_utterances = []
        for line in full_transcript.split('\n'):
            if line.strip().lower().startswith('patient:'):
                utterance = line.split(':', 1)[1].strip()
                if utterance:
                    patient_utterances.append(utterance)

        if not patient_utterances:
            return {"message": "No patient dialogue found in the transcript."}

        # 2. Focus on the last two utterances for richer final context
        final_context = " ".join(patient_utterances[-7:-1])

        print(f"Analyzing final patient context: '{final_context}'")

        # 3. Classify the combined final context for sentiment and intent
        sentiment_result = self.classifier(final_context, self.sentiment_labels, multi_label=False)
        intent_result = self.classifier(final_context, self.intent_labels, multi_label=False)

        final_analysis = {
            "sentiment": sentiment_result['labels'][0],
            "intent": intent_result['labels'][0],
            "context analyzed": final_context
        }

        return final_analysis