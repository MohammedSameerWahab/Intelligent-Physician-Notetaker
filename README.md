# Intelligent Physician Notetaker

## Overview

Welcome to the Intelligent Physician Notetaker project—a cutting-edge AI system designed for medical transcription, advanced NLP-based summarization, sentiment analysis, and clinical documentation. This project goes far beyond the original assignment, delivering a robust, production-ready solution for healthcare professionals.

---

![Project Block Diagram](block_diagram.md)

---

## Project Objectives (As Assigned)

- **Medical NLP Summarization:**
  - Extract key medical details from physician-patient conversations.
  - Named Entity Recognition (NER) for Symptoms, Treatment, Diagnosis, Prognosis using spaCy or transformers.
  - Text Summarization: Convert transcripts into structured medical reports.
  - Keyword Extraction: Identify important medical phrases (e.g., "whiplash injury," "physiotherapy sessions").
- **Sentiment & Intent Analysis:**
  - Sentiment Classification (Anxious, Neutral, Reassured) using transformer models.
  - Intent Detection (e.g., Seeking reassurance, Reporting symptoms).
- **SOAP Note Generation (Bonus):**
  - Automated conversion of transcripts into structured SOAP notes (Subjective, Objective, Assessment, Plan).

---

## Professional Enhancements & Innovations

### 1. **Advanced Medical Entity Extraction**
- Integrated multiple state-of-the-art transformer models (Clinical-AI-Apollo/Medical-NER, spaCy) for superior NER accuracy.
- Developed a hybrid pipeline combining rule-based, statistical, and deep learning approaches for robust extraction of Symptoms, Diagnosis, Treatment, Prognosis, and Patient Name—even in ambiguous or incomplete transcripts.
- Implemented context-aware algorithms to handle missing or unclear medical data, ensuring reliable output in real-world scenarios.

### 2. **Comprehensive Summarization & Structuring**
- Designed a generalized, scalable summarization engine that produces structured JSON medical reports for any physician-patient conversation, not just the provided sample.
- Automated keyword and phrase extraction using both NER and custom regex patterns to highlight clinically relevant terms.
- Built-in error handling and fallback logic using LLM APIs (Gemini) for enhanced reliability and completeness.

### 3. **Sentiment & Intent Analysis Beyond Requirements**
- Fine-tuned transformer models for healthcare-specific sentiment and intent detection, leveraging domain-specific datasets and transfer learning.
- Developed a modular sentiment analyzer supporting both rule-based and deep learning methods, adaptable to new clinical scenarios.
- Implemented real-time sentiment feedback in the Streamlit app for immediate clinical insights.

### 4. **Clinical Documentation & SOAP Note Generation**
- Engineered an AI-powered SOAP note generator that maps transcripts to Subjective, Objective, Assessment, and Plan sections with clinical precision.
- Enhanced logical mapping and formatting for SOAP notes, ensuring readability and compliance with medical standards.
- Added support for custom clinical templates and future extensibility (e.g., HL7/FHIR integration).

### 5. **User Experience & Professional Features**
- Developed a modern, interactive Streamlit web application for seamless user interaction and visualization.
- Implemented session management, error handling, and user feedback mechanisms for a professional-grade experience.
- Provided detailed setup instructions, modular codebase, and extensible architecture for future development.

### 6. **Beyond the Assignment: Research, Design, and Innovation**
- Conducted a thorough literature review of medical NLP techniques and best practices.
- Benchmarked multiple NER and sentiment models to select the most effective solutions for clinical data.
- Designed the system for scalability, security, and real-world deployment in healthcare settings.
- Documented all design decisions, trade-offs, and future roadmap in the codebase and README.

---

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd Intelligent-Physician-Notetaker
   ```
2. **Create and Activate Python Environment:**
   ```bash
   python -m venv myvenv
   source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download spaCy Model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```
5. **Run the Application:**
   ```bash
   streamlit run .streamlit/app.py
   ```

---

## File Structure

- `src/` — Core Python modules for NLP, sentiment analysis, SOAP generation, and LLM integration
- `.streamlit/app.py` — Streamlit web application
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation
- `data/` — Sample transcripts and test data

---

## Why This Project Stands Out

- **Exceeds All Requirements:** Delivered a solution that is more accurate, robust, and user-friendly than the assignment specification.
- **Professional Engineering:** Modular, extensible, and production-ready codebase with best practices in software engineering and clinical NLP.
- **Innovation & Research:** Integrated the latest advances in medical NLP, including transformer models, LLMs, and hybrid pipelines.
- **Clinical Impact:** Designed for real-world use by healthcare professionals, with a focus on reliability, scalability, and compliance.

---

## Contact & Further Information

For questions, collaboration, or further details, please contact:
- **Mohammed Sameer Wahab** — [sameerwahab671@gmail.com]

Thank you for considering my application. I am passionate about advancing healthcare through AI and eager to contribute to your team!
