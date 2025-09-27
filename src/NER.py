from transformers import pipeline
import spacy
from typing import Dict, List, Optional
import re

def extract_medical_entities(text: str) -> Dict:
    """
    Extract medical entities from a physician-patient conversation.
    
    Args:
        text (str): The conversation transcript
    
    Returns:
        Dict: Structured medical information
    """
    # Load spaCy model for general NER
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize medical NER pipeline
    medical_ner = pipeline("ner", model="Clinical-AI-Apollo/Medical-NER")
    
    # Initialize result structure
    result = {
        "Patient_Name": None,
        "Symptoms": [],
        "Diagnosis": None,
        "Treatment": [],
        "Current_Status": None,
        "Prognosis": None
    }
    name_match = re.search(r'(?:Mr|Ms|Mrs)\.\s*(\w+)', text, re.IGNORECASE)
    if name_match:
        # Reconstruct the full title and name found in the text.
        result["Patient_Name"] = name_match.group(0)
    else:
        # Strategy 2 (Fallback): Look for PERSON entities in the physician's lines,
        # as the physician is the one who will address the patient by name.
        physician_lines = [line for line in text.split('\n') if line.strip().startswith('Physician:')]
        for line in physician_lines:
            line_doc = nlp(line)
            for ent in line_doc.ents:
                if ent.label_ == "PERSON":
                    result["Patient_Name"] = ent.text
                    # Break out of all loops once the first potential name is found.
                    break
            if result["Patient_Name"]:
                break
    
    # Process text with spaCy for general entities
    doc = nlp(text)
    
    # Extract patient name (looking for PERSON entities in patient's responses)
    patient_lines = [line for line in text.split('\n') if line.strip().startswith('Patient:')]
    for line in patient_lines:
        line_doc = nlp(line)
        for ent in line_doc.ents:
            if ent.label_ == "PERSON":
                result["Patient_Name"] = ent.text
                break
    
    # Use medical NER for medical entities
    medical_entities = medical_ner(text)
    
    # Process medical entities
    for entity in medical_entities:
        entity_text = entity['word']
        entity_type = entity['entity']
        
        # Map entity types to our structure
        if 'SYMPTOM' in entity_type or 'PROBLEM' in entity_type:
            if entity_text not in result["Symptoms"]:
                result["Symptoms"].append(entity_text)
        elif 'TREATMENT' in entity_type or 'PROCEDURE' in entity_type:
            if entity_text not in result["Treatment"]:
                result["Treatment"].append(entity_text)
        elif 'DIAGNOSIS' in entity_type or 'DISEASE' in entity_type:
            if not result["Diagnosis"]:
                result["Diagnosis"] = entity_text
    
    # Extract current status and prognosis using context analysis
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'current' in line.lower() or 'now' in line.lower():
            result["Current_Status"] = line.split(':', 1)[1].strip() if ':' in line else line.strip()
        if 'expect' in line.lower() or 'prognosis' in line.lower() or 'outlook' in line.lower():
            result["Prognosis"] = line.split(':', 1)[1].strip() if ':' in line else line.strip()
    
    # Clean up empty values
    result = {k: v for k, v in result.items() if v}
    
    return result

def process_medical_conversation(transcript_path: str) -> Dict:
    """
    Process a medical conversation and return structured information.
    
    Args:
        transcript_path (str): Path to the transcript file
        
    Returns:
        Dict: Structured medical information
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return extract_medical_entities(text)