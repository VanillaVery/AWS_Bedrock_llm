import torch
# from datasets import load_dataset
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from transformers import AutoTokenizer
# from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
AWS Bedrock Runtime API example using Anthropic Claude model.
1. AWS_PROFILE_NAME 입력
2. user_content 입력
3. (옵션)assistant_content 입력
"""
import boto3
import json
import logging

from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

AWS_PROFILE_NAME = "poc"
user_content = "아직 그녀는 나를 좋아하지 않아서 너무 대놓고 사랑에 대한 책을 선물하기에는 부담스러워."
user_message = {"role": "user", "content": user_content}
assistant_content = "30대 남자이고 독서 모임에서 맘에 드는 분이 생기셨군요, 그 분도 당신을 좋아하나요?"
assistant_message = {"role": "assistant", "content": assistant_content}

system_prompt = """
Variables:

{'$SITUATION'}

************************

Prompt:
당신은 사용자의 특정 상황에 기반한 책 추천과 설명을 제공할 것입니다.

먼저, {$SITUATION}에 제공된 사용자의 상황을 주의 깊게 읽고 이해하세요. 그들의 상황에 대한 세부사항을 완전히 파악하기 위해 필요하다면 명확히 하는 질문을 하세요.

사용자의 상황을 잘 이해하게 되면, 그들에게 가장 관련성이 있고 도움이 될 만한 책을 생각해보세요. 상황에서 반영된 사용자의 목표, 도전, 관심사 또는 인생의 단계와 같은 요소들을 고려하세요.

적절한 책 추천을 선택한 후에는, 그 이유를 자세히 설명하세요. 구체적으로, 추천하는 책의 내용, 주제 또는 교훈이 사용자의 상황과 어떻게 일치하고 이를 해결할 수 있는지 설명하세요. 이 책이 좋은 선택이 될 이유에 대한 사려 깊은 정당화를 제공하세요.

마지막으로, 책 추천과 그것을 선택한 이유에 대한 설명을 제공하세요. 다음과 같은 형식으로 응답을 구성하세요:

<book_recommendation>
[책 제목] [저자]
</book_recommendation>

<explanation>
[책의 내용과 사용자의 필요/목표/도전과 어떻게 일치하는지에 기반한 사용자의 상황에 대한 좋은 추천 이유에 대한 자세한 설명.]
</explanation>

사용자의 상황에 대해 추천을 제공하기 전에 명확히 해야 할 것이 있다면, 추가 질문을 해주세요. 사용자의 필요를 이해하는 것을 보여주는 철저하고 논리적인 응답을 제공하세요.
"""


def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens):

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
    )

    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get("body").read())

    return response_body


def main():
    """
    Entrypoint for Anthropic Claude message example.
    """

    try:

        session = boto3.Session(profile_name=AWS_PROFILE_NAME)
        bedrock_runtime = session.client(service_name="bedrock-runtime")

        # model_id = "anthropic.claude-v2:1"
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        max_tokens = 1000

        # Prompt with user turn only.
        messages = [user_message]

        response = generate_message(
            bedrock_runtime, model_id, system_prompt, messages, max_tokens
        )
        print("User turn only.")
        print(json.dumps(response, indent=4))

        # Prompt with both user turn and prefilled assistant response.
        # Anthropic Claude continues by using the prefilled assistant text.
        messages = [user_message, assistant_message]
        response = generate_message(
            bedrock_runtime, model_id, system_prompt, messages, max_tokens
        )
        print("User turn and prefilled assistant response.")
        print(json.dumps(response, indent=4))

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))


if __name__ == "__main__":
    main()