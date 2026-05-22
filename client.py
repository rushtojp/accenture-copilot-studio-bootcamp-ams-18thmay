import os
import asyncio
import json
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputParam

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Clear the console
os.system('cls' if os.name == 'nt' else 'clear')

# Load environment variables from .env file
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

async def connect_to_server(exit_stack: AsyncExitStack):
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=None
    )

    # Start the MCP server
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport

    # Create an MCP client session
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()

    # List available tools
    response = await session.list_tools()
    tools = response.tools
    print("
Connected to server with tools:", [tool.name for tool in tools])

    return session

async def chat_loop(session):
    # Connect to the agents client
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        # Get the mcp tools available from the server
        response = await session.list_tools()
        tools = response.tools

        # Build a function for each tool
        def make_tool_func(tool_name):
            async def tool_func(**kwargs):
                result = await session.call_tool(tool_name, kwargs)
                return result
            tool_func.__name__ = tool_name
            return tool_func

        # Store the functions in a dictionary for easy access
        functions_dict = {tool.name: make_tool_func(tool.name) for tool in tools}

        # Create FunctionTool definitions for the agent
        mcp_function_tools = []
        for tool in tools:
            function_tool = FunctionTool(
                name=tool.name,
                description=tool.description,
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=True
            )
            mcp_function_tools.append(function_tool)

        # Create the agent
        agent = project_client.agents.create_version(
            agent_name="inventory-agent",
            definition=PromptAgentDefinition(
                model=model_deployment,
                instructions="""
                    You are an inventory assistant. Here are some general guidelines:
                    - Recommend restock if item inventory < 10 and weekly sales > 15
                    - Recommend clearance if item inventory > 20 and weekly sales < 5
                """,
                tools=mcp_function_tools
            ),
        )

        # Create a thread for the chat session
        conversation = openai_client.conversations.create()

        while True:
            user_input = input("Enter a prompt for the inventory agent. Use 'quit' to exit.\nUSER: ").strip()
            if user_input.lower() == "quit":
                print("Exiting chat.")
                break

            # Create an input list to hold function call outputs for each turn
            input_list: ResponseInputParam = []

            # Send a prompt to the agent
            openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[{"type": "message", "role": "user", "content": user_input}],
            )

            # Retrieve the agent's response
            response = openai_client.responses.create(
                conversation=conversation.id,
                extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
                input=input_list,
            )

            # Check for failures
            if response.status == "failed":
                print(f"Response failed: {response.error}")
                continue

            # Process function calls if the model requests them
            for item in response.output:
                if item.type == "function_call":
                    function_name = item.name
                    kwargs = json.loads(item.arguments)
                    required_function = functions_dict.get(function_name)

                    if required_function:
                        output = await required_function(**kwargs)
                        
                        input_list.append(
                            FunctionCallOutput(
                                type="function_call_output",
                                call_id=item.call_id,
                                output=output.content[0].text,
                            )
                        )

            # Send function call outputs back to the model if any were collected
            if input_list:
                response = openai_client.responses.create(
                    input=input_list,
                    previous_response_id=response.id,
                    extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
                )
            
            # Print the final text output
            if hasattr(response, 'output_text'):
                print(f"Agent response: {response.output_text}")
            else:
                # Fallback to extract text from output items
                for item in response.output:
                    if item.type == "message" and item.role == "assistant":
                        for content in item.content:
                            if content.type == "text":
                                print(f"Agent response: {content.text}")

        # Delete the agent when done
        print("Cleaning up agents:")
        project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
        print("Deleted inventory agent.")

async def main():
    exit_stack = AsyncExitStack()
    try:
        session = await connect_to_server(exit_stack)
        await chat_loop(session)
    finally:
        await exit_stack.aclose()

if __name__ == "__main__":
    asyncio.run(main())
