# src/services/insights.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    """Get API key from environment variables or Streamlit secrets."""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["GEMINI_API_KEY"]
        except ImportError:
            # Streamlit not available (running outside Streamlit app)
            pass
        except KeyError:
            # Key not found in Streamlit secrets
            pass
    
    return api_key

# Configure once with your API key
genai.configure(api_key=get_api_key())


def query_gemini_flash(modality: str, label: str, confidence: float) -> str:
    """
    Calls Gemini Flash to generate a clinician-friendly explanation
    for a given classification result.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        You are an AI medical assistant. A clinician uploaded a {modality} scan.
        The automated model classified it as: {label}
        with confidence {confidence:.2%}.

        Please provide a short, plain-language insight:
        - Describe what this finding means clinically.
        - Mention next steps or cautions, but keep it concise.
        - Use professional but clear tone.
        """

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"(Gemini insight unavailable: {e})"
