from src.agent import RAGAgent

agent = RAGAgent()

print(
    agent.answer(
        "What was 3M's worldwide net sales in Q2 2023?"
    )
)

print(
    agent.answer(
        "What page was that information on?"
    )
)