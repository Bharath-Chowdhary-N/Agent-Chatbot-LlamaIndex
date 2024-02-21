import numpy as np
from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import netherlands_engine
# Loads the.env file
load_dotenv()

population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(df=population_df, verbose=True)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
population_query_engine.query("What is the population of Canada?")

tools = [
    note_engine, 
    QueryEngineTool(query_engine=population_query_engine,metadata=ToolMetadata(name="pop_data",description="about world population")),
    QueryEngineTool(query_engine=netherlands_engine,metadata=ToolMetadata(name="dutch_data",description="about netherlands")),
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quiit): ")) != "q":
    result = agent.query(prompt)
    print(result)