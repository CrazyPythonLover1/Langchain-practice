import os
import getpass
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import streamlit as st 
# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader('./', glob="ifarmacia.mdf", loader_kwargs=text_loader_kwargs)

docs = loader.load()
print(len(docs))

raw_documents = TextLoader('./ProdutosFarmacia.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

# query = "What did the president say about Ketanji Brown Jackson"
# docs = db.similarity_search(query)
# print(docs[0].page_content, " 29 search ")

doc_sources = [doc.metadata['source']  for doc in docs]
print(doc_sources)

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
