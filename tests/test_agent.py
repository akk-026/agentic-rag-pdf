from src.agent import RAGAgent
from src.db import create_chat, init_db

init_db()
chat_id = create_chat("Test")

agent = RAGAgent()

print(agent.answer("Give me a summary of Curiflow.", chat_id))
print(agent.answer("What roles do they offer?", chat_id))