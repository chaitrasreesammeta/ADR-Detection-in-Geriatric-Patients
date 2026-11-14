from config import load_config
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_config()

def run():
    model = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        HumanMessage(content="What is the difference between language R and python?"),
        HumanMessage(content="What is language R.")
    ]
    response = model.invoke(input=messages)
    print(response.content)

if __name__ == '__main__':
    run()