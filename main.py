import os
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

import streamlit as st 

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


st.title('LangChain Demo with OpenAI API')
text_input = st.text_input('Search the topic u want')

# Prompt Template 
template = """\
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate.from_template(template)
# prompt.format(product="colorful socks")

llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm, prompt=prompt, verbose=True)


if text_input:
  st.write(chain.run(text_input))
