from config import load_config
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_config()

def main():
    model = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        SystemMessage(content="You are a helpful assistant and responds to the question in a funny way"),
        HumanMessage(content="Who is the hero in Bahubali movie")
    ]
    response = model.invoke(messages)
    print(response.content)

if __name__ == '__main__':
    main()