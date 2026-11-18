from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
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

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

def display_answer(result):
    messages = result.get("messages", [])

    # Find the last AI message containing text
    answer = next(
        (
            msg.content
            for msg in reversed(messages)
            if msg.type == "ai"
            and isinstance(msg.content, str)
            and msg.content.strip()
        ),
        None
    )

    if not answer:
        console.print("[yellow]No answer found.[/yellow]")
        return

    console.print()

    console.print(
        Panel(
            Markdown(answer),
            title="[bold cyan]🤖 AI Assistant[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            expand=True
        )
    )

    console.print()


def main():
    print("Hello from langchain course!")
    result = agent.invoke({ "messages":[
        # HumanMessage(content="What is the weather in Tokyo?")
        HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in London on each of these platform: linkedin, indeed, Reead.co.uk and Handshake and list their details")
    ] })

    # Get the final AI response
    display_answer(result)
    print(result)

if __name__ == "__main__":
    main()