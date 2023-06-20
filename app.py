import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import openai
import streamlit as st

openai.api_key = os.environ["OPENAI_API_KEY"]

documents = SimpleDirectoryReader('./data').load_data()

max_input_size = 4096
num_outputs = 512
max_chunk_overlap = 20
chunk_size_limit = 600

prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio=max_chunk_overlap/100.0, chunk_size_limit=chunk_size_limit)
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

custom_LLM_index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

def query_index(query):
    query_engine = custom_LLM_index.as_query_engine()
    response = query_engine.query(query)
    return response

def main():
    st.title("Tarka Chatbot")
    st.markdown("Write a headline: What makes Tarka unique?")
    text_input = st.text_input("Enter your query:")
    if st.button("Submit"):
        response = query_index(text_input)
        st.text_area("Response:", value=response)

if __name__ == '__main__':
    main()