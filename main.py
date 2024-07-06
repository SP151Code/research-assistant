from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.schema.runnable import RunnableLambda
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import json
import streamlit as st

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


RESULTS_PER_QUESTION = 3
ddg_search = DuckDuckGoSearchAPIWrapper()

def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]

summary_template = """
{text}

---------------------------
Using the above text, answer in short the following question:

> {question}
---------------------------
If the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats, etc if available.
"""

summary_prompt = ChatPromptTemplate.from_template(summary_template)

def scrape_text(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator = " ", strip = True)
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

url = "https://blog.langchain.dev/announcing-langsmith/"

scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
    text = lambda x: scrape_text(x["url"])[:10000]
    ) | summary_prompt | ChatOpenAI(model = "gpt-4o") | StrOutputParser()
) | (lambda x: f"URL : {x['url']}\n\nSummary:\n\n{x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"]),
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            """
            Write 3 google search queries to search online that form an objective
            opinion from the following: {question}\n
            You must respond with a list of strings in the following format:
            [["query1"], ["query2"], ["query3"]]
            """,
        ),
    ]
)

search_question_chain = search_prompt | ChatOpenAI(model = "gpt-4o") | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: list(map(lambda y: {"question": y[0]}, x))) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

RESEARCH_REPORT_TEMPLATE = """
Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary =  full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model = "gpt-4o") | StrOutputParser()