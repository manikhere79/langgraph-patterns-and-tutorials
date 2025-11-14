"""
001 - Basic State in LangGraph

This tutorial introduces the fundamental concept of STATE in LangGraph.
State is how your graph maintains and passes information between nodes.

Key Concepts:
- TypedDict for defining state schema
- State annotations and reducers
- How state flows through the graph

Learning Objectives:
1. Understand what state is and why it matters
2. Define a state schema using TypedDict
3. Use Annotated for state reducers
4. See how state is updated and passed between nodes
"""

import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# ============================================================================
# CONCEPT 1: Defining State with TypedDict
# ============================================================================
# State in LangGraph is like a shared memory that flows through your graph.
# Each node can read from it and write to it. Think of it as a dictionary
# that gets passed around and updated at each step.


class CustomerSupportState(TypedDict):
    """
    State schema for our customer support system.
    
    Each field represents a piece of information we want to track
    throughout the conversation.
    """
    # User's message - what they're asking about
    user_message: str
    
    # List of all messages in the conversation
    # Annotated with operator.add means new messages get APPENDED to the list
    messages: Annotated[list[str], operator.add]
    
    # Current step in the workflow (for tracking)
    current_step: str
    
    # User's name (if we know it)
    customer_name: str


# ============================================================================
# CONCEPT 2: Creating Nodes (Functions that process state)
# ============================================================================
# Nodes are functions that take the current state and return updates to it.
# They don't modify state directly - they return a dictionary of updates.


def greet_customer(state: CustomerSupportState) -> dict:
    """
    First node: Greet the customer
    
    Args:
        state: Current state of the conversation
        
    Returns:
        Dictionary with state updates
    """
    print(f"\n{'='*60}")
    print(f"NODE: Greet Customer")
    print(f"{'='*60}")
    print(f"Current state received: {state}")
    
    # Extract customer name from state
    name = state.get("customer_name", "there")
    
    # Create greeting message
    greeting = f"Hello {name}! Welcome to Customer Support. How can I help you today?"
    
    # Return state updates
    # The 'messages' field uses operator.add, so this will APPEND
    return {
        "messages": [greeting],
        "current_step": "greeted"
    }


def acknowledge_message(state: CustomerSupportState) -> dict:
    """
    Second node: Acknowledge the user's message
    
    Args:
        state: Current state with user's message
        
    Returns:
        Dictionary with state updates
    """
    print(f"\n{'='*60}")
    print(f"NODE: Acknowledge Message")
    print(f"{'='*60}")
    print(f"User message: {state['user_message']}")
    
    # Create acknowledgment
    acknowledgment = f"I understand you're asking about: '{state['user_message']}'"
    
    return {
        "messages": [acknowledgment],
        "current_step": "acknowledged"
    }


def provide_response(state: CustomerSupportState) -> dict:
    """
    Third node: Provide a simple response
    
    Args:
        state: Current state
        
    Returns:
        Dictionary with state updates
    """
    print(f"\n{'='*60}")
    print(f"NODE: Provide Response")
    print(f"{'='*60}")
    
    # Simple canned response
    response = "Thank you for reaching out. A support agent will assist you shortly."
    
    return {
        "messages": [response],
        "current_step": "responded"
    }


# ============================================================================
# CONCEPT 3: Building the Graph
# ============================================================================
# A graph connects nodes together in a specific flow.
# StateGraph manages the state as it flows through nodes.


def create_basic_graph():
    """
    Create a simple linear graph:
    START -> Greet -> Acknowledge -> Respond -> END
    """
    # Initialize the graph with our state schema
    graph = StateGraph[CustomerSupportState, None, CustomerSupportState, CustomerSupportState](CustomerSupportState)
    
    # Add nodes to the graph
    # Each node is a function that processes state
    graph.add_node("greet", greet_customer)
    graph.add_node("acknowledge", acknowledge_message)
    graph.add_node("respond", provide_response)
    
    # Define the flow: how nodes connect to each other
    graph.add_edge(START, "greet")          # Start with greeting
    graph.add_edge("greet", "acknowledge")  # Then acknowledge
    graph.add_edge("acknowledge", "respond") # Then respond
    graph.add_edge("respond", END)          # Finally, end
    
    # Compile the graph into a runnable object
    return graph.compile()


# ============================================================================
# CONCEPT 4: Running the Graph
# ============================================================================


def run_example():
    """
    Run the basic state example
    """
    print("\n" + "="*60)
    print("🚀 EXAMPLE: Basic State in LangGraph")
    print("="*60)
    
    # Create the graph
    app = create_basic_graph()
    
    # Define initial state
    initial_state = {
        "user_message": "I need help with my order",
        "messages": [],  # Start with empty message list
        "current_step": "started",
        "customer_name": "Alice"
    }
    
    print("\n📝 Initial State:")
    print(f"   User: {initial_state['customer_name']}")
    print(f"   Message: {initial_state['user_message']}")
    
    # Run the graph
    # The state flows through: greet -> acknowledge -> respond
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "="*60)
    print("✅ FINAL STATE")
    print("="*60)
    print(f"\nCurrent Step: {final_state['current_step']}")
    print(f"\n📨 Conversation Messages:")
    for i, message in enumerate(final_state['messages'], 1):
        print(f"   {i}. {message}")
    
    print("\n" + "="*60)
    print("💡 KEY TAKEAWAYS")
    print("="*60)
    print("1. State is a TypedDict that flows through the graph")
    print("2. Annotated[list, operator.add] appends items to lists")
    print("3. Nodes return dictionaries to update state")
    print("4. State accumulates as it flows through nodes")
    print("="*60 + "\n")


# ============================================================================
# Additional Example: Understanding State Reducers
# ============================================================================


def demonstrate_reducers():
    """
    Show the difference between regular fields and fields with reducers
    """
    print("\n" + "="*60)
    print("🔧 BONUS: Understanding State Reducers")
    print("="*60)
    
    class SimpleState(TypedDict):
        # Regular field - gets REPLACED each time
        counter: int
        # Field with operator.add - gets ADDED each time
        total: Annotated[int, operator.add]
        # List with operator.add - items get APPENDED
        items: Annotated[list[str], operator.add]
    
    def node1(state: SimpleState) -> dict:
        return {
            "counter": 5,      # Sets counter to 5
            "total": 10,       # Adds 10 to total
            "items": ["A"]     # Appends "A" to items
        }
    
    def node2(state: SimpleState) -> dict:
        return {
            "counter": 3,      # REPLACES counter with 3 (not 5+3)
            "total": 20,       # ADDS 20 to total (10+20=30)
            "items": ["B"]     # APPENDS "B" to items
        }
    
    graph = StateGraph(SimpleState)
    graph.add_node("node1", node1)
    graph.add_node("node2", node2)
    graph.add_edge(START, "node1")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", END)
    
    app = graph.compile()
    
    initial = {"counter": 0, "total": 0, "items": []}
    final = app.invoke(initial)
    
    print(f"\n📊 Results:")
    print(f"   counter: {final['counter']} (replaced, not added)")
    print(f"   total: {final['total']} (accumulated: 0+10+20)")
    print(f"   items: {final['items']} (appended)")
    
    print("\n💡 Reducer Types:")
    print("   - No annotation: REPLACES value")
    print("   - operator.add: ADDS/APPENDS values")
    print("="*60 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Run the main example
    run_example()
    
    # Run the bonus reducer demonstration
    demonstrate_reducers()
    
    print("\n🎓 Next Steps:")
    print("   Run 002-simple-chatbot.py to learn about integrating LLMs")
    print("   with state management!\n")


