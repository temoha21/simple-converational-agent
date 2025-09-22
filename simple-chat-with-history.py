from langchain_core.runnables import RunnableConfig
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY')


llm = GoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=1000, temperature=0)
#simple in memory storage for message history
store = {}

def get_chat_history(user_id):
    if user_id not in store:
        store[user_id] = ChatMessageHistory()
    return store[user_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps people find information."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
])

chain = prompt | llm # Returns RunnableSequence object

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

session_id = input("Enter uniq user name: ")  # Example user/session identifier

while True:
    query = input("Enter your question: ")
    if query.lower() in ['exit', 'quit']:
        break
    if query.lower() in ['logout']:
        print(f"Logging out user: {session_id}")
        session_id = input("Enter new uniq user name: ")
        print(f"Logged in as new user: {session_id}")
        continue

    response = chain_with_history.invoke({"input": query}, config=RunnableConfig(configurable={
        "session_id": session_id
    }))
    print("AI:", response)