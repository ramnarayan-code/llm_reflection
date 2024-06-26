# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c8xKBR7egKG3YGa5Zw9_jATktBskEmib
"""

# Commented out IPython magic to ensure Python compatibility.
#  %pip install -U --quiet  langchain langchain-openai

import os
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = ''
print(os.getenv('OPENAI_API_KEY'))

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a python code generator."
            " Generate the best code in terms of space and time complexity."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
llm = ChatOpenAI()
generate = prompt | llm

code = ""
request = HumanMessage(
    content="Write a program for string palindrome:"
)
for chunk in generate.stream({"messages": [request]}):
    print(chunk.content, end="")
    code += chunk.content

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a critique assessing the program. Generate critique and recommendations for the user's submission."
            " Provide detailed recommendations, including time complexity and space complexity",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm

reflection = ""
for chunk in reflect.stream({"messages": [request, HumanMessage(content=code)]}):
    print(chunk.content, end="")
    reflection += chunk.content

for chunk in generate.stream(
    {"messages": [request, AIMessage(content=code), HumanMessage(content=reflection)]}
):
    print(chunk.content, end="")