import json

import requests
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from rich import print as pprint

from model_configurations import get_model_configuration

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)


class Event(BaseModel):
    date: str
    name: str


class ResponseFormat(BaseModel):
    Result: list[Event]


class Hw3ResultItem(BaseModel):
    add: bool
    reason: str


class Hw3ResponseFormat(BaseModel):
    Result: Hw3ResultItem


@tool
def get_holidays_tool(year, month):
    """
    get taiwan holidays from calendarific api
    :param year:
    :param month:
    :return:
    """
    url = "https://calendarific.com/api/v2/holidays"
    params = {
        "api_key": "0ekqUNenCqQDyksWEE9tGjjY7nPUrFqI",
        "country": "TW",
        "year": year,
        "month": month
    }
    response = requests.get(url, params=params)
    holidays = response.json().get("response", {}).get("holidays", [])
    return holidays


llm = AzureChatOpenAI(
    model=gpt_config['model_name'],
    deployment_name=gpt_config['deployment_name'],
    openai_api_key=gpt_config['api_key'],
    openai_api_version=gpt_config['api_version'],
    azure_endpoint=gpt_config['api_base'],
    temperature=gpt_config['temperature']
)
prompt = hub.pull("hwchase17/openai-functions-agent")
tools = [get_holidays_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
json_parser = JsonOutputParser()


def generate_hw01(question):
    output_parser = PydanticOutputParser(pydantic_object=ResponseFormat)
    format_instructions = output_parser.get_format_instructions()
    message = llm.invoke(question + f"{format_instructions}")
    output = json_parser.invoke(message)
    json_output = json.dumps(output)
    # pprint(output)
    return json_output


def generate_hw02(question):
    output_parser = PydanticOutputParser(pydantic_object=ResponseFormat)
    format_instructions = output_parser.get_format_instructions()
    message = executor.invoke({"input": question + f"{format_instructions}"})
    output = json_parser.invoke(message['output'])
    json_output = json.dumps(output)
    print(json_output)
    return json_output


def generate_hw03(question2, question3):
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    output_parser = PydanticOutputParser(pydantic_object=Hw3ResponseFormat)
    format_instructions = output_parser.get_format_instructions()
    agent_with_chat_history = RunnableWithMessageHistory(
        executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    agent_with_chat_history.invoke({"input": question2}, config={"configurable": {"session_id": "<foo>"}})
    message = agent_with_chat_history.invoke(
        {"input": question3 + f"{format_instructions}" + "如果不存在，則add為 true；否則add為 false。"},
        config={"configurable": {"session_id": "<foo>"}})
    output = json_parser.invoke(message['output'])
    json_output = json.dumps(output, ensure_ascii=False)
    pprint(json_output)
    return json_output


def generate_hw04(question):
    pass


def demo(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )
    response = llm.invoke([message])

    return response
