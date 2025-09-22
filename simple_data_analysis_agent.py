import os

from langchain.agents import AgentType
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY')
agent_scratchpad = """
Human: {query}
AI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.

Action: python_repl_ast

Action Input: 
"""
# Set a random seed for reproducibility
np.random.seed(42)
# Generate sample data
n_rows = 1000

def get_agent_scratchpad(user_question: str) -> str:
    return agent_scratchpad.format(query=user_question)

llm = GoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=1000, temperature=0)

# Generate dates
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_rows)]

# Define data categories
makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

# Create the dataset
data = {
    'Date': dates,
    'Make': np.random.choice(makes, n_rows),
    'Model': np.random.choice(models, n_rows),
    'Color': np.random.choice(colors, n_rows),
    'Year': np.random.randint(2015, 2023, n_rows),
    'Price': np.random.uniform(20000, 80000, n_rows).round(2),
    'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
    'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
    'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
    'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
}

# Create DataFrame and sort by date
df = pd.DataFrame(data).sort_values('Date')

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

while True:
    query = input("Enter your question: ")
    if query.lower() in ['exit', 'quit']:
        break

    answer = agent.invoke({
        "input": query,
        "agent_scratchpad": get_agent_scratchpad(query),
    })
    print("AI:", answer.get('output', 'No answer'))
    print("-------")