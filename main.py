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
If the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats, etc.
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
    text = lambda x: scrape_text(x["url"])[:10000]
    ) | summary_prompt | ChatOpenAI(model = "gpt-4o") | StrOutputParser()

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

chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

chain.invoke(
    {
        "question": "What is the difference between Langsmith and Langchain?",
    }
)