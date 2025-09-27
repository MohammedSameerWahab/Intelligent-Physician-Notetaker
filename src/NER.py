from transformers import pipeline
import spacy
from typing import Dict, List, Optional
import re

def extract_medical_entities(text: str) -> Dict:
    """
    Extracts and cleans medical entities using a more robust hybrid approach:
    1. A specialized NER model for clinical entities (Symptoms, Treatments).
    2. A Question-Answering (QA) model for contextual information (Status, Prognosis).
    
    Args:
        text (str): The conversation transcript
    
    Returns:
        Dict: Structured and cleaned medical information
    """
    # --- 1. Initialization ---
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize NER pipeline for specific entities
    ner_pipeline = pipeline(
        "ner", 
        model="Clinical-AI-Apollo/Medical-NER",
        aggregation_strategy="simple" 
    )
    
    # NEW: Initialize QA pipeline for contextual, sentence-level information
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )
    
    stop_words = {
        'patient', 'doctor', 'physician', 'examination', 'physical', 'activity', 
        'stethoscope', 'oxygen', 'fluids', 'up', 's', 'of', 'in', 'the', 'a', 'to'
    }

    result = {
        "Patient_Name": None,
        "Symptoms": [],
        "Diagnosis": None,
        "Treatment": [],
        "Current_Status": "Not mentioned",
        "Prognosis": "Not mentioned"
    }

    # --- 2. Patient Name Extraction (Unchanged) ---
    name_match = re.search(r'(?:Mr|Ms|Mrs)\.\s*(\w+)', text, re.IGNORECASE)
    if name_match:
        result["Patient_Name"] = name_match.group(0)
    else:
        physician_lines = [line for line in text.split('\n') if line.strip().startswith('Physician:')]
        for line in physician_lines:
            line_doc = nlp(line)
            for ent in line_doc.ents:
                if ent.label_ == "PERSON":
                    result["Patient_Name"] = ent.text
                    break
            if result["Patient_Name"]:
                break
                
    # --- 3. Medical Entity Extraction via NER (Unchanged) ---
    ner_entities = ner_pipeline(text)
    
    for entity in ner_entities:
        entity_text = entity['word'].strip().lower()
        entity_type = entity['entity_group']
        
        if len(entity_text) <= 1 or entity_text in stop_words:
            continue

        if 'SYMPTOM' in entity_type or 'PROBLEM' in entity_type:
            if entity_text not in result["Symptoms"]:
                result["Symptoms"].append(entity_text)
        elif 'TREATMENT' in entity_type or 'PROCEDURE' in entity_type:
            if entity_text not in result["Treatment"]:
                result["Treatment"].append(entity_text)
        elif 'DIAGNOSIS' in entity_type or 'DISEASE' in entity_type:
            if not result["Diagnosis"]:
                result["Diagnosis"] = entity_text
    
    # --- 4. ENHANCED Context-Based Extraction via QA ---
    # This replaces the brittle keyword search with a robust QA model.
    
    # Question for Current Status
    status_question = "What is the patient's current status or how are they feeling now?"
    status_answer = qa_pipeline(question=status_question, context=text)
    if status_answer['score'] > 0.1: # Confidence threshold
        result["Current_Status"] = status_answer['answer'].strip()

    # Question for Prognosis
    prognosis_question = "What is the physician's prognosis or expectation for recovery?"
    prognosis_answer = qa_pipeline(question=prognosis_question, context=text)
    if prognosis_answer['score'] > 0.1: # Confidence threshold
        result["Prognosis"] = prognosis_answer['answer'].strip()
    
    # --- 5. Final Cleanup (Unchanged) ---
    if not result["Symptoms"]:
        result["Symptoms"] = ["Not mentioned"]
    if not result["Treatment"]:
        result["Treatment"] = ["Not mentioned"]
    if not result["Diagnosis"]:
        result["Diagnosis"] = "Not mentioned"

    return result

def process_medical_conversation(transcript_text: str) -> Dict:
    """
    Process a medical conversation and return structured information.
    
    Args:
        transcript_text (str): The raw text of the transcript
        
    Returns:
        Dict: Structured medical information
    """
    return extract_medical_entities(transcript_text)

