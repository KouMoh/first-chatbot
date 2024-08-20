import dotenv
import google.generativeai as genai
import os

from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()


review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

review_chain = review_prompt_template | model





#
# import dotenv
# import google.generativeai as genai
# import os
#
#
# dotenv.load_dotenv()
#
# genai.configure(api_key=os.environ["API_KEY"])
#
# model = genai.GenerativeModel('gemini-1.5-flash')
#
# print(os.environ.get("API_KEY"))
