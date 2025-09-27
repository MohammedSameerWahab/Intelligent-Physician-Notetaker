# src/soap_generator.py

import os
import json
import google.generativeai as genai
from typing import Dict
import streamlit as st

def generate_soap_note(transcript: str) -> Dict:
    """
    Generates a structured SOAP note from a medical transcript using the Gemini Pro model.

    Args:
        transcript: The full text of the medical conversation.

    Returns:
        A dictionary representing the structured SOAP note.
    """
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not found in Streamlit Secrets."}

    genai.configure(api_key=api_key)

    # --- Prompt Engineering: This is the core instruction for the LLM ---
    prompt = f"""
    You are an expert medical scribe with years of experience in clinical documentation. Your task is to analyze the following physician-patient transcript and convert it into a structured SOAP note.

    **SOAP Note Definitions:**
    - **Subjective (S):** The patient's personal account of their symptoms and history. Includes the Chief Complaint and History of Present Illness.
    - **Objective (O):** Measurable, quantifiable, and observable data from the physician's examination. Includes Physical Exam findings and direct observations.
    - **Assessment (A):** The physician's diagnosis or impression of the patient's condition.
    - **Plan (P):** The proposed course of action, including treatments, medications, and follow-up instructions.

    **Instructions:**
    1.  Read the entire transcript carefully.
    2.  Logically map the conversational details into the S, O, A, and P sections.
    3.  The output MUST be a single, valid JSON object with the exact nested structure shown below. Do not add any text or formatting outside of this JSON object.

    **Expected JSON Format:**
    {{
      "Subjective": {{
        "Chief_Complaint": "A concise summary of the patient's main reason for the visit.",
        "History_of_Present_Illness": "A detailed narrative of the patient's symptoms and the story of their illness."
      }},
      "Objective": {{
        "Physical_Exam": "Findings from the physician's physical examination.",
        "Observations": "General observations about the patient's appearance and condition."
      }},
      "Assessment": {{
        "Diagnosis": "The official diagnosis given by the physician.",
        "Severity": "The assessed severity of the condition (e.g., Mild, Moderate, Severe, Improving)."
      }},
      "Plan": {{
        "Treatment": "Specific treatments or therapies planned.",
        "Follow-Up": "Instructions for patient follow-up visits."
      }}
    }}

    **Transcript to Analyze:**
    ---
    {transcript}
    ---

    **Generated SOAP Note (JSON Output):**
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Temperature 0.0 makes the output deterministic and less creative.
        generation_config = genai.types.GenerationConfig(temperature=0.0)
        
        print("\nGenerating SOAP note with Gemini...")
        response = model.generate_content(prompt, generation_config=generation_config)

        # Clean the response to ensure it's a parsable JSON string
        cleaned_response = response.text.strip().lstrip("```json").rstrip("```")
        
        soap_note_json = json.loads(cleaned_response)
        return soap_note_json

    except json.JSONDecodeError:
        return {
            "error": "Gemini did not return a valid JSON object.",
            "raw_output": response.text if 'response' in locals() else "No response from model."
        }
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}