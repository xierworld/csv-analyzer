import pandas as pd
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from prompt import prompt_template
import json
# from dotenv import load_dotenv, find_dotenv
# import pandas
# import os

# load_dotenv(find_dotenv())

def dataframe_agent(dashscope_api_key, df, query):
    model = ChatTongyi(model="qwen-max", dashscope_api_key=dashscope_api_key)

    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent_executor_kwargs={"handle_parsing_errors": True},
        verbose=True)

    prompt = prompt_template + query
    response = agent.invoke({"input": prompt})
    response_dict = json.loads(response["output"])
    return response_dict

# df = pd.read_csv("personal_data.csv")
# print(dataframe_agent(os.getenv("DASHSCOPE_API_KEY"), df, "文件里有几行数据"))