import os
from langchain.llms import bedrock
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
from langchain.llms import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator, FAISS
from langchain_community.document_loaders import DataFrameLoader


# 다른 모듈과 같이 맞물려 동작할 수 있게 해주는 모듈


# 후에 시스템 프롬프트를 구성
# kwargs 추가...
def bedrock_chatbot():
    llm = BedrockChat(
        credentials_profile_name="poc",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"temperature": 0.5},
    )

    return llm


def get_memory():
    # 입력과 출력을 저장
    # buff_memory = bedrock_chatbot()
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
    return memory



def get_rag_chat_response(input_text, memory, index): #chat client 함수
    
    llm = bedrock_chatbot()
    
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(), memory=memory)
    
    chat_response = conversation_with_retrieval({"question": input_text}) #사용자 메시지, 기록 및 지식을 모델에 전달합니다.
    
    return chat_response['answer']


def get_index():
    # 예제 데이터 (책 제목과 소개 포함)
    book_data = pd.DataFrame({
        'title': ['Book Title 1', 'Book Title 2', ...],  # 책 제목
        'description': ['Description 1', 'Description 2', ...]  # 책 소개
    })

    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"),
    )

    # 각 책의 내용을 로드하기 위한 커스텀 로더 생성
    loaders = [DataFrameLoader(text=row['title'] + " " + row['description']) for _, row in book_data.iterrows()]

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
    )

    index_from_loader = index_creator.from_loaders(loaders)

    return index_from_loader

index = get_index()
