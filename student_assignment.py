import base64
import json
from mimetypes import guess_type

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
from openai import AzureOpenAI
from pydantic import BaseModel
from rich import print as pprint

from model_configurations import get_model_configuration

# Constants and configurations
gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
PARSER = JsonOutputParser()


# Define Pydantic Models
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


class Hw4ResultItem(BaseModel):
    score: int


class Hw4ResponseFormat(BaseModel):
    Result: Hw4ResultItem


# Define Tool
@tool
def get_holidays_tool(year, month):
    """
    Get Taiwan holidays from Calendarific API.
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


# Initialize LLM
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


def local_image_to_data_url(image_path):
    """
    Convert a local image file to a data URL.
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


def generate_hw01(question):
    """
    Answer a question using the AzureChatOpenAI model.
    """
    output_parser = PydanticOutputParser(pydantic_object=ResponseFormat)
    format_instructions = output_parser.get_format_instructions()
    message = llm.invoke(question + f"{format_instructions}")
    output = json_parser.invoke(message)
    return json.dumps(output)


def generate_hw02(question):
    """
    Answer a question using an agent executor with additional tools.
    """
    output_parser = PydanticOutputParser(pydantic_object=ResponseFormat)
    format_instructions = output_parser.get_format_instructions()
    message = executor.invoke({"input": question + f"{format_instructions}"})
    output = json_parser.invoke(message['output'])
    return json.dumps(output)


def generate_hw03(question2, question3):
    """
    Manage chat history and handle multi-step questions.
    """
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
        config={"configurable": {"session_id": "<foo>"}}
    )
    output = json_parser.invoke(message['output'])
    return json.dumps(output, ensure_ascii=False)


def generate_hw04(question):
    """
    Answer a question based on a provided image file.
    """
    api_base = gpt_config['api_base']
    deployment_name = gpt_config['deployment_name']
    client = AzureOpenAI(
        api_key=gpt_config['api_key'],
        api_version="2023-12-01-preview",
        base_url=f"{api_base}/openai/deployments/{deployment_name}"
    )
    data_url = local_image_to_data_url("baseball.png")
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ],
        max_tokens=2000
    )
    output_parser = PydanticOutputParser(pydantic_object=Hw4ResponseFormat)
    format_instructions = output_parser.get_format_instructions()
    message = llm.invoke(question + response.json() + f"{format_instructions}")
    output = json_parser.invoke(message)
    return json.dumps(output, ensure_ascii=False)


def demo(question):
    """
    Demonstrate a simple question handling with LLM.
    """
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )
    response = llm.invoke([message])
    return response
