import os

from dotenv import load_dotenv
import google.generativeai as genai

from src.llm.base import BaseLLM

load_dotenv()


class GeminiProvider(BaseLLM):
    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Put it in your .env file."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text or ""