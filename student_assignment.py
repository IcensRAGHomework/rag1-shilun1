import json
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
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


def generate_hw01(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )
    json_parser = JsonOutputParser()
    output_parser = PydanticOutputParser(pydantic_object=ResponseFormat)
    format_instructions = output_parser.get_format_instructions()
    message = llm.invoke(question + f"{format_instructions}")
    json_output = json_parser.invoke(message)
    # pprint(json_output)
    return json.dumps(json_output)


def generate_hw02(question):
    pass


def generate_hw03(question2, question3):
    pass


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
