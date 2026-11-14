from urllib import response

from config import load_config
from langchain_openai import ChatOpenAI

load_config()

def main():
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(input="crack me a joke.")
    print(response.content)


if __name__ == "__main__":
    main()
