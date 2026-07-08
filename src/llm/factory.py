from src.llm.gemini_provider import get_gemini_llm
from src.llm.groq_provider import get_groq_llm

LLM_PROVIDER = "groq"
# LLM_PROVIDER = "gemini"


def get_llm():
    if LLM_PROVIDER == "groq":
        return get_groq_llm()

    return get_gemini_llm()