from src.agent import RAGAgent

agent = RAGAgent()

response = agent.answer("What are the three main HIPAA rules?")
print(response)