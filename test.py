import os
from langchain.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage


# 호출 테스트
def bedrock_chatbot(input_text):
    chat = BedrockChat(
        credentials_profile_name="poc",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"temperature": 0.5},
    )
    messages = [HumanMessage(content=input_text)]
    return chat(messages)


bedrock_chatbot("대한민국에서 가장 유명한 위인을 말해줘")
