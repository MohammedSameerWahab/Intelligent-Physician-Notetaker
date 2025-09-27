graph TB
    subgraph "Input Layer"
        A[Audio/Text Input] --> B[Speech-to-Text Transcription]
        B --> C[Raw Transcript]
    end

    subgraph "NLP Processing Layer"
        C --> D[Medical NER Pipeline]
        C --> E[Sentiment Analysis]
        C --> F[SOAP Note Generator]
        
        subgraph "Medical NER Pipeline"
            D --> D1[Clinical-AI-Apollo/Medical-NER]
            D --> D2[spaCy NER]
            D1 --> D3[Entity Extraction]
            D2 --> D3
            D3 --> D4[Context Analysis]
        end
        
        subgraph "Sentiment Analysis"
            E --> E1[Transformer Model]
            E --> E2[Rule-Based Analysis]
            E1 --> E3[Sentiment Classification]
            E2 --> E3
        end
        
        subgraph "SOAP Generator"
            F --> F1[Subjective Extraction]
            F --> F2[Objective Analysis]
            F --> F3[Assessment Processing]
            F --> F4[Plan Generation]
        end
    end
    
    subgraph "Backup & Enhancement"
        G[Google Gemini LLM]
        D4 --> G
        E3 --> G
        F4 --> G
    end
    
    subgraph "Output Layer"
        D4 --> H[Structured Medical Summary]
        E3 --> I[Sentiment & Intent Report]
        F4 --> J[SOAP Documentation]
        G --> K[Enhanced Analysis]
        
        H --> L[Web Interface]
        I --> L
        J --> L
        K --> L
    end
    
    subgraph "UI Layer (Streamlit Web Application)"
        L --> M[Interactive Dashboard]
        L --> N[Results Visualization]
        L --> O[Download Options]
    end

    style "Input Layer" fill:#e6f3ff,stroke:#333,stroke-width:2px
    style "NLP Processing Layer" fill:#f0fff0,stroke:#333,stroke-width:2px
    style "Backup & Enhancement" fill:#fff0f0,stroke:#333,stroke-width:2px
    style "Output Layer" fill:#fff0e6,stroke:#333,stroke-width:2px
    style "UI Layer (Streamlit Web Application)" fill:#f0f0ff,stroke:#333,stroke-width:2px
    
    classDef pipeline fill:#f9f,stroke:#333,stroke-width:2px
    class D1,D2,D3,D4,E1,E2,E3,F1,F2,F3,F4 pipeline
