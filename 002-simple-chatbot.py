"""
002 - Simple Chatbot with LangGraph + Google Generative AI

This tutorial shows how to integrate a Large Language Model (LLM) into your
LangGraph application. We'll build a simple customer support chatbot.

Key Concepts:
- Integrating Google Generative AI (Gemini)
- Using LLMs within graph nodes
- Formatting messages for LLMs
- Streaming responses

Learning Objectives:
1. Connect Google Generative AI to LangGraph
2. Create an LLM-powered node
3. Handle conversation context
4. Format state for LLM consumption
"""

import os
from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# ============================================================================
# CONCEPT 1: Setting up Google Generative AI
# ============================================================================
# Google's Gemini models are powerful, fast, and have a generous free tier.
# We'll use the ChatGoogleGenerativeAI class from langchain.


def create_llm():
    """
    Create and configure the Google Generative AI model
    
    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Stable, widely available model (gemini-2.5-flash doesn't exist)
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,  # Controls randomness (0=focused, 1=creative)
        max_output_tokens=1024,  # Maximum response length
    )


# ============================================================================
# CONCEPT 2: State Schema for Chatbot
# ============================================================================
# We need to track conversation history and metadata


class ChatbotState(TypedDict):
    """
    State for our chatbot application
    """
    # List of messages (will be LangChain message objects)
    messages: Annotated[list, add]
    
    # User's current input
    user_input: str
    
    # AI's last response (for easy access)
    ai_response: str
    
    # Conversation metadata
    conversation_id: str
    turn_count: int


# ============================================================================
# CONCEPT 3: Creating LLM-Powered Nodes
# ============================================================================


def chatbot_node(state: ChatbotState) -> dict:
    """
    Main chatbot node that processes user input with LLM
    
    This node:
    1. Takes the user's input from state
    2. Adds it to the conversation history
    3. Sends the conversation to the LLM
    4. Returns the AI's response
    
    Args:
        state: Current conversation state
        
    Returns:
        State updates with AI response
    """
    print(f"\n{'='*60}")
    print("🤖 CHATBOT NODE - Processing with LLM")
    print(f"{'='*60}")
    
    # Get user input
    user_input = state.get("user_input", "")
    print(f"👤 User: {user_input}")
    
    # Create LLM instance
    llm = create_llm()
    
    # Build message history
    # Start with a system message to set the chatbot's behavior
    messages = state.get("messages", [])
    
    # If this is the first message, add system prompt
    if not messages:
        system_prompt = SystemMessage(content="""
You are a helpful and friendly customer support agent for TechShop, 
an electronics retailer. Your role is to:

- Greet customers warmly
- Answer questions about orders, products, and policies
- Be empathetic and professional
- Escalate to human agents when needed
- Keep responses concise but helpful

Current company policies:
- Free shipping on orders over $50
- 30-day return policy
- 24/7 customer support
- Price match guarantee
        """)
        messages.append(system_prompt)
    
    # Add the user's message
    messages.append(HumanMessage(content=user_input))
    
    # Call the LLM
    print("🔄 Calling Google Generative AI...")
    response = llm.invoke(messages)
    
    # Extract the AI's response
    ai_response = response.content
    print(f"🤖 AI: {ai_response}")
    
    # Update state
    return {
        "messages": [HumanMessage(content=user_input), AIMessage(content=ai_response)],
        "ai_response": ai_response,
        "turn_count": state.get("turn_count", 0) + 1
    }


def summary_node(state: ChatbotState) -> dict:
    """
    Optional node to summarize the conversation
    
    Args:
        state: Current conversation state
        
    Returns:
        State updates with summary
    """
    print(f"\n{'='*60}")
    print("📊 SUMMARY NODE")
    print(f"{'='*60}")
    
    turn_count = state.get("turn_count", 0)
    print(f"Conversation turns: {turn_count}")
    print(f"Messages exchanged: {len(state.get('messages', []))}")
    
    return {}


# ============================================================================
# CONCEPT 4: Building the Chatbot Graph
# ============================================================================


def create_chatbot_graph():
    """
    Create a simple chatbot graph:
    START -> Chatbot -> Summary -> END
    """
    # Initialize graph
    graph = StateGraph[ChatbotState, None, ChatbotState, ChatbotState](ChatbotState)
    
    # Add nodes
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("summary", summary_node)
    
    # Define flow
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", "summary")
    graph.add_edge("summary", END)
    
    return graph.compile()


# ============================================================================
# CONCEPT 5: Running the Chatbot
# ============================================================================


def run_single_turn():
    """
    Example: Single turn conversation
    """
    print("\n" + "="*60)
    print("💬 EXAMPLE 1: Single Turn Conversation")
    print("="*60)
    
    app = create_chatbot_graph()
    
    # Initial state with user message
    initial_state = {
        "user_input": "Hi! I want to know about your return policy.",
        "messages": [],
        "ai_response": "",
        "conversation_id": "conv_001",
        "turn_count": 0
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*60)
    print("✅ CONVERSATION COMPLETE")
    print("="*60)
    print(f"Turns: {final_state['turn_count']}")
    print(f"\nFinal Response:\n{final_state['ai_response']}")
    print("="*60 + "\n")


def run_multi_turn():
    """
    Example: Multi-turn conversation
    """
    print("\n" + "="*60)
    print("💬 EXAMPLE 2: Multi-Turn Conversation")
    print("="*60)
    
    app = create_chatbot_graph()
    
    # Simulate multiple turns
    conversation = [
        "Hello! I need help with my order.",
        "My order number is #12345. Can you check its status?",
        "When will it arrive?",
        "Thank you for your help!"
    ]
    
    state = {
        "messages": [],
        "conversation_id": "conv_002",
        "turn_count": 0,
        "user_input": "",
        "ai_response": ""
    }
    
    for i, user_message in enumerate(conversation, 1):
        print(f"\n{'─'*60}")
        print(f"TURN {i}")
        print(f"{'─'*60}")
        
        # Update state with new user input
        state["user_input"] = user_message
        
        # Run the graph
        state = app.invoke(state)
        
        # Print the exchange
        print(f"\n👤 User: {user_message}")
        print(f"🤖 AI: {state['ai_response']}")
    
    print("\n" + "="*60)
    print("✅ MULTI-TURN CONVERSATION COMPLETE")
    print("="*60)
    print(f"Total turns: {state['turn_count']}")
    print(f"Total messages: {len(state['messages'])}")
    print("="*60 + "\n")


# ============================================================================
# BONUS: Streaming Responses
# ============================================================================


def streaming_chatbot_node(state: ChatbotState) -> dict:
    """
    Chatbot node with streaming responses
    
    Streaming shows the response as it's generated, token by token.
    This provides better UX for longer responses.
    
    Includes error handling and fallback to non-streaming if streaming fails.
    """
    print(f"\n{'='*60}")
    print("🌊 STREAMING CHATBOT NODE")
    print(f"{'='*60}")
    
    user_input = state.get("user_input", "")
    print(f"👤 User: {user_input}\n🤖 AI: ", end="", flush=True)
    
    # Create LLM instance
    llm = create_llm()
    
    # Build messages
    messages = state.get("messages", [])
    if not messages:
        messages.append(SystemMessage(content="You are a helpful customer support agent."))
    messages.append(HumanMessage(content=user_input))
    
    # Try streaming with error handling and fallback
    full_response = ""
    try:
        chunk_count = 0
        for chunk in llm.stream(messages):
            # Check if chunk has content
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                print(content, end="", flush=True)
                full_response += content
                chunk_count += 1
        
        # If no chunks were received, fallback to invoke
        if chunk_count == 0:
            print("\n⚠️  Streaming returned no chunks, falling back to non-streaming...", end="", flush=True)
            response = llm.invoke(messages)
            full_response = response.content
            print(full_response)
    
    except ValueError as e:
        # Handle "No generation chunks were returned" error
        if "No generation chunks" in str(e):
            print("\n⚠️  Streaming failed, using non-streaming fallback...", end="", flush=True)
            response = llm.invoke(messages)
            full_response = response.content
            print(full_response)
        else:
            raise  # Re-raise other ValueError exceptions
    
    except Exception as e:
        # Handle any other streaming errors with fallback
        print(f"\n⚠️  Streaming error: {str(e)[:100]}, falling back to non-streaming...", end="", flush=True)
        response = llm.invoke(messages)
        full_response = response.content
        print(full_response)
    
    print()  # New line after streaming/response
    
    return {
        "messages": [HumanMessage(content=user_input), AIMessage(content=full_response)],
        "ai_response": full_response,
        "turn_count": state.get("turn_count", 0) + 1
    }


def demo_streaming():
    """
    Demonstrate streaming responses
    """
    print("\n" + "="*60)
    print("💬 EXAMPLE 3: Streaming Responses")
    print("="*60)
    
    # Create graph with streaming node
    graph = StateGraph(ChatbotState)
    graph.add_node("chatbot", streaming_chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    app = graph.compile()
    
    # Run with a question that generates a longer response
    state = {
        "user_input": "Can you explain your entire return and refund process in detail?",
        "messages": [],
        "conversation_id": "conv_003",
        "turn_count": 0,
        "ai_response": ""
    }
    
    final_state = app.invoke(state)
    
    print("\n" + "="*60)
    print("✅ STREAMING COMPLETE")
    print("="*60 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please check your .env file and make sure it contains your API key")
        exit(1)
    
    # Run examples
    run_single_turn()
    run_multi_turn()
    demo_streaming()
    
    print("\n" + "="*60)
    print("💡 KEY TAKEAWAYS")
    print("="*60)
    print("1. LLMs are integrated as regular Python objects")
    print("2. Use SystemMessage to set chatbot behavior")
    print("3. HumanMessage and AIMessage track conversation")
    print("4. Streaming provides better UX for long responses")
    print("5. State carries conversation history between turns")
    print("="*60)
    
    print("\n🎓 Next Steps:")
    print("   Run 003-tools-integration.py to learn about adding")
    print("   tools to your chatbot!\n")


