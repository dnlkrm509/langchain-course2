from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from langchain.tools import tool
# from tavily import TavilyClient
from langchain_tavily import TavilySearch


### What I did earlier I wrote a custom search, instead it's a far better job
## if I use a built-in tavily search tool because tavily team wrote a package (langchain-tavily)
##  which takes the Tavily SDK and wraps it as a langchain tool
## tavily team has written much better description and arguments.

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#          query: The query is search for
#          Returns:
#              The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the question")
    sources:List[Source] = Field(default_factory=list, description="A list of sources used by the agent to generate the answer")


llm = ChatOpenAI(model="gpt-5")
tools = [
    TavilySearch(
        max_results=3
    )
] # I limited the tavily search result Because it was expensive, and I'm learning agents
print(tools[0].name)
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
    system_prompt="""
    You are a professional job search assistant.

    When searching for jobs:
    1. Find relevant and currently available job postings.
    2. Prioritize jobs explicitly mentioning LangChain or LangGraph.
    3. Include job title, company, location, responsibilities,
       required skills, experience level, and salary when available.
    4. Include the original job posting URL.
    5. Never invent job details or salary information.
    6. Clearly indicate when information is unavailable.
    7. Format the answer using clean Markdown.
    8. Include the URLs in the sources field.
    """,
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="tavily_search",
            run_limit=10,
            exit_behavior="end"
        )
    ]
)


from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def display_answer(result):
    structured = result.get("structured_response")

    if not structured:
        console.print(
            Panel(
                "[bold red]No structured response found.[/bold red]",
                title="Error",
                border_style="red"
            )
        )
        return

    # Support Pydantic objects and dictionaries
    if isinstance(structured, BaseModel):
        answer = structured.answer
        sources = structured.sources
    else:
        answer = structured.get("answer", "")
        sources = structured.get("sources", [])

    if not answer:
        console.print("[yellow]The agent returned an empty answer.[/yellow]")
        return

    console.print()

    # Display answer
    console.print(
        Panel(
            Markdown(answer),
            title="[bold cyan]🤖 AI Assistant[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            expand=True
        )
    )

    # Display sources
    if sources:
        table = Table(
            title="📚 Sources",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            expand=True
        )

        table.add_column("#", style="dim", width=4)
        table.add_column("Source URL", style="cyan", overflow="fold")

        seen_urls = set()

        for source in sources:
            url = source.url if isinstance(source, BaseModel) else source.get("url", "")

            if url and url not in seen_urls:
                seen_urls.add(url)
                table.add_row(
                    str(len(seen_urls)),
                    f"[link={url}]{url}[/link]"
                )

        console.print(table)

    console.print()


def main():
    print("Hello from langchain course!")
    result = agent.invoke({ "messages":[
        # HumanMessage(content="What is the weather in Tokyo?")
        # HumanMessage(
        #     content=(
        #         "Search for 3 job postings for an ai engineer "
        #         "using langchain in London on each of this platform: "
        #         "linkedin"
        #         # "linkedin, indeed, Reead.co.uk and Handshake. " # this search is too expensive
        #         "list their details."
        #     )
        # )

        HumanMessage(
            content=(
                "Find 3 AI Engineer job postings in London "
                "on LinkedIn that explicitly mention LangChain. "
                "Return the job title, company, location, "
                "requirements, responsibilities, salary if available, "
                "and original LinkedIn URL. "
                "Do not search other platforms."
            )
        )
    ] })

    print(type(result["structured_response"]))

    # Get the final AI response
    display_answer(result)
    # print(result)

if __name__ == "__main__":
    main()