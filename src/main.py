# src/main.py

import os
import json
from gemini_llm import get_gemini_summary
from NER import process_medical_conversation
from sentiment_analyzer import SentimentAnalyzer
from soap_generator import generate_soap_note

def main():
    """
    Main function to run the hybrid summarization pipeline with enhanced NER.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcript_path = os.path.join(base_dir, 'data', 'transcript.txt')

    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found at {transcript_path}")
        return

    # --- Tier 1: Medical Entity Recognition ---
    medical_summary = process_medical_conversation(transcript_path)
    
    print("\n--- Tier 1: Structured Medical Summary ---")
    print(json.dumps(medical_summary, indent=2, ensure_ascii=False))
    print("-----------------------------------------\n")

# --- Sentiment & Intent Analysis ---
    analyzer = SentimentAnalyzer()
    sentiment_analysis = analyzer.analyze_final_state(transcript_path)
    print("\n--- Patient Sentiment & Intent Analysis ---")
    print(json.dumps(sentiment_analysis, indent=2, ensure_ascii=False))
    print("-------------------------------------------\n")


    # --- Tier 2: Optional Gemini Summary ---
    user_choice = input("Would you like an advanced AI summary from Gemini? (y/n): ")

    if user_choice.lower() == 'y':
        # Read the transcript content to pass to Gemini
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_content = f.read()
            
        gemini_summary = get_gemini_summary(transcript_content)
        
        print("\n--- Tier 2: Advanced Gemini Summary ---")
        print(json.dumps(gemini_summary, indent=2, ensure_ascii=False))
        print("---------------------------------------\n")
    else:
        print("Exiting.")

    user_choice = input("Would you like to generate a SOAP note with Gemini? (y/n): ")

    if user_choice.lower() == 'y':
        # Read the transcript content to pass to the generator
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_content = f.read()
            
        soap_note = generate_soap_note(transcript_content)
        
        print("\n--- Step 3: AI-Generated SOAP Note ---")
        print(json.dumps(soap_note, indent=2, ensure_ascii=False))
        print("--------------------------------------\n")
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()