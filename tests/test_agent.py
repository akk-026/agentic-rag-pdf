from src.agent import RAGAgent

agent = RAGAgent()

response = agent.answer( "What was 3M's worldwide net sales in the second quarter of 2023?")
print(response)