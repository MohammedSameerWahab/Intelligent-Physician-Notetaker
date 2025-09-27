# src/gemini_summarizer.py

import os
import json
import google.generativeai as genai
from typing import Dict

def get_gemini_summary(transcript: str) -> Dict:
    """
    Generates a structured medical summary using the Gemini Pro model.

    Args:
        transcript: The full text of the medical conversation.

    Returns:
        A dictionary containing the structured summary.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY environment variable not set."}

    genai.configure(api_key=api_key)

    # This prompt is critical. It gives the LLM clear instructions and constraints.
    prompt = f"""
    Act as an expert medical scribe. Your task is to analyze the following medical transcript and extract key information into a structured JSON format.

    **Instructions:**
    1.  Read the entire transcript carefully.
    2.  Identify the patient's name, reported symptoms, the physician's diagnosis, treatments mentioned, the patient's current status, and the final prognosis.
    3.  Format the output as a single, valid JSON object with these exact keys: "Patient_Name", "Symptoms", "Diagnosis", "Treatment", "Current_Status", "Prognosis".
    4.  The "Symptoms" and "Treatment" fields MUST be lists of strings.
    5.  If a piece of information is not explicitly mentioned, use the string "Not mentioned".

    **Transcript:**
    ---
    {transcript}
    ---

    **JSON Output:**
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Set temperature=0 for deterministic, fact-based output
        generation_config = genai.types.GenerationConfig(temperature=0.0)
        
        print("\nAsking Gemini for an advanced summary...")
        response = model.generate_content(prompt, generation_config=generation_config)

        # Clean up the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        
        summary_json = json.loads(cleaned_response)
        return summary_json

    except json.JSONDecodeError:
        return {
            "error": "Gemini did not return a valid JSON object.",
            "raw_output": response.text if 'response' in locals() else "No response from model."
        }
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}