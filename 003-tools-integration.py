"""
003 - Tools Integration in LangGraph

This tutorial introduces TOOLS - functions that LLMs can call to perform actions
or retrieve information. We'll build a customer support chatbot with FAQ search.

Key Concepts:
- Defining tools with @tool decorator
- Tool binding to LLMs
- Tool calling and execution
- Integrating tool results into conversation

Learning Objectives:
1. Create custom tools for your chatbot
2. Bind tools to LLMs
3. Handle tool calls and responses
4. Build a tool-enabled conversation flow
"""

import os
from typing import TypedDict, Annotated, Literal
from operator import add
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# ============================================================================
# CONCEPT 1: Creating Tools
# ============================================================================
# Tools are Python functions that LLMs can call. The @tool decorator
# converts them into LangChain tools with proper schemas.


@tool
def search_faq(query: str) -> str:
    """
    Search the FAQ database for answers to common questions.
    
    Use this tool when customers ask about:
    - Shipping policies
    - Return policies
    - Payment methods
    - Store hours
    - General product information
    
    Args:
        query: The customer's question or search query
        
    Returns:
        The relevant FAQ answer if found, or a message indicating no match
    """
    # Simulated FAQ database
    # In production, this would query a real database or vector store
    faq_database = {
        "shipping": """
        SHIPPING POLICY:
        - Free standard shipping on orders over $50
        - Standard shipping (5-7 business days): $5.99
        - Express shipping (2-3 business days): $15.99
        - Overnight shipping: $25.99
        - International shipping available to select countries
        - Track your order at www.techshop.com/tracking
        """,
        
        "return": """
        RETURN POLICY:
        - 30-day return window from date of delivery
        - Items must be unused and in original packaging
        - Free return shipping for defective items
        - Refunds processed within 5-7 business days
        - Exchanges available for different sizes/colors
        - Contact support@techshop.com to initiate a return
        """,
        
        "payment": """
        PAYMENT METHODS:
        - All major credit cards (Visa, MasterCard, Amex, Discover)
        - PayPal
        - Apple Pay and Google Pay
        - Buy Now, Pay Later options available
        - Gift cards and store credit accepted
        - Secure checkout with SSL encryption
        """,
        
        "hours": """
        STORE HOURS:
        - Monday - Friday: 9:00 AM - 9:00 PM EST
        - Saturday: 10:00 AM - 8:00 PM EST
        - Sunday: 11:00 AM - 6:00 PM EST
        - Online store: Open 24/7
        - Customer support: 24/7 via chat and email
        - Phone support: Monday-Friday 8:00 AM - 10:00 PM EST
        """,
        
        "warranty": """
        WARRANTY INFORMATION:
        - All products come with manufacturer's warranty
        - Extended warranty plans available at checkout
        - Warranty period varies by product (typically 1-3 years)
        - Covers manufacturing defects and hardware failures
        - Register your product for warranty coverage
        - Claims can be filed online or via customer support
        """
    }
    
    # Simple keyword matching (in production, use semantic search)
    query_lower = query.lower()
    
    for topic, answer in faq_database.items():
        if topic in query_lower:
            print(f"   ✅ Found FAQ match for: {topic}")
            return answer
    
    # Check for common keywords
    keyword_mapping = {
        "ship": "shipping",
        "deliver": "shipping",
        "refund": "return",
        "exchange": "return",
        "pay": "payment",
        "credit card": "payment",
        "open": "hours",
        "close": "hours",
        "guarantee": "warranty",
        "broken": "warranty"
    }
    
    for keyword, topic in keyword_mapping.items():
        if keyword in query_lower:
            print(f"   ✅ Found FAQ match via keyword '{keyword}': {topic}")
            return faq_database[topic]
    
    print(f"   ℹ️  No FAQ match found for query: {query}")
    return "I couldn't find a specific FAQ answer. Let me help you in another way or connect you with a specialist."


@tool
def check_order_status(order_number: str) -> str:
    """
    Check the status of a customer's order.
    
    Use this when customers ask about:
    - Order status
    - Delivery updates
    - Tracking information
    - Order confirmation
    
    Args:
        order_number: The customer's order number (format: #12345)
        
    Returns:
        Current order status and tracking information
    """
    # Simulated order database
    # In production, this would query your order management system
    orders = {
        "#12345": {
            "status": "In Transit",
            "shipped_date": "2024-01-15",
            "expected_delivery": "2024-01-20",
            "tracking": "1Z999AA10123456784",
            "carrier": "UPS"
        },
        "#12346": {
            "status": "Processing",
            "shipped_date": None,
            "expected_delivery": "2024-01-22",
            "tracking": None,
            "carrier": "USPS"
        },
        "#12347": {
            "status": "Delivered",
            "shipped_date": "2024-01-10",
            "expected_delivery": "2024-01-14",
            "tracking": "9405511899223147823456",
            "carrier": "USPS"
        }
    }
    
    order = orders.get(order_number)
    
    if not order:
        return f"I couldn't find order {order_number}. Please check the order number and try again, or contact support for assistance."
    
    print(f"   ✅ Found order: {order_number}")
    
    status_msg = f"""
    ORDER STATUS for {order_number}:
    
    Status: {order['status']}
    Carrier: {order['carrier']}
    """
    
    if order['shipped_date']:
        status_msg += f"Shipped: {order['shipped_date']}\n    "
    
    if order['tracking']:
        status_msg += f"Tracking #: {order['tracking']}\n    "
    
    status_msg += f"Expected Delivery: {order['expected_delivery']}"
    
    return status_msg


@tool
def create_support_ticket(issue_description: str, priority: Literal["low", "medium", "high"] = "medium") -> str:
    """
    Create a support ticket for issues that need human attention.
    
    Use this when:
    - Customer has a complex issue
    - Issue requires investigation
    - Customer requests human support
    - Tools can't resolve the issue
    
    Args:
        issue_description: Description of the customer's issue
        priority: Ticket priority (low, medium, high)
        
    Returns:
        Ticket confirmation with ticket number
    """
    # Generate a ticket number (in production, this would be from your ticketing system)
    import random
    ticket_number = f"TICKET-{random.randint(10000, 99999)}"
    
    print(f"   🎫 Created support ticket: {ticket_number} (Priority: {priority})")
    
    return f"""
    Support ticket created successfully!
    
    Ticket Number: {ticket_number}
    Priority: {priority.upper()}
    Status: Open
    
    A support specialist will review your case and respond within:
    - High priority: 2 hours
    - Medium priority: 24 hours
    - Low priority: 48 hours
    
    You'll receive email updates at each step.
    """


# ============================================================================
# CONCEPT 2: State Schema with Tool Calls
# ============================================================================


class ToolEnabledState(TypedDict):
    """
    State for chatbot with tool capabilities
    """
    # Conversation messages (includes tool calls and results)
    messages: Annotated[list, add]
    
    # User's current input
    user_input: str
    
    # Track if tools were used
    tools_used: Annotated[list[str], add]


# ============================================================================
# CONCEPT 3: Creating Tool-Enabled Nodes
# ============================================================================


def create_llm_with_tools():
    """
    Create an LLM with tools bound to it
    
    Returns:
        ChatGoogleGenerativeAI instance with tools
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )
    
    # Bind tools to the LLM
    # This tells the LLM what tools are available and how to use them
    tools = [search_faq, check_order_status, create_support_ticket]
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools


def chatbot_with_tools(state: ToolEnabledState) -> dict:
    """
    Chatbot node that can call tools
    
    This node:
    1. Receives user input
    2. Decides if tools are needed
    3. Calls appropriate tools
    4. Returns response (with or without tool calls)
    """
    print(f"\n{'='*60}")
    print("🤖 CHATBOT WITH TOOLS")
    print(f"{'='*60}")
    
    # Get messages from state
    messages = state.get("messages", [])
    
    # Determine if this is a new conversation turn or responding to tool results
    last_message = messages[-1] if messages else None
    is_responding_to_tools = isinstance(last_message, ToolMessage)
    
    # Track messages we need to add to state
    messages_to_add = []
    
    # Add system message ONLY on the very first turn (when messages is empty)
    # This ensures SystemMessage is only added once and maintains proper Gemini message sequence
    if not messages:
        system_prompt = SystemMessage(content="""
You are a helpful customer support agent for TechShop.

You have access to tools to help customers:
1. search_faq: Search our FAQ database for policy information
2. check_order_status: Look up order status and tracking
3. create_support_ticket: Create tickets for complex issues

Always:
- Use tools when you need specific information
- Be friendly and professional
- Explain what you're doing when using tools
- Escalate complex issues by creating tickets
        """)
        messages = [system_prompt]
        messages_to_add.append(system_prompt)
    
    # Add user message ONLY if this is a new user turn (not responding to tool results)
    # When responding to tools, messages already contain the full conversation history
    user_input = state.get("user_input", "")
    if user_input and not is_responding_to_tools:
        # Ensure we don't already have this user message
        is_last_message_user = isinstance(last_message, HumanMessage)
        if not is_last_message_user:
            print(f"👤 User: {user_input}")
            user_message = HumanMessage(content=user_input)
            messages.append(user_message)
            messages_to_add.append(user_message)
    
    # Call LLM with tools
    # Always bind tools - Gemini will decide if they're needed based on the message sequence
    llm_with_tools = create_llm_with_tools()
    print("🔄 Calling LLM with tools...")
    
    # Debug: Print message sequence for troubleshooting
    if is_responding_to_tools:
        print(f"   📋 Responding to tool results. Message count: {len(messages)}")
        print(f"   📋 Last message type: {type(last_message).__name__}")
        print(f"   📋 Message types in sequence: {[type(m).__name__ for m in messages]}")
    
    response = llm_with_tools.invoke(messages)
    
    # Check if LLM wants to call tools
    if response.tool_calls:
        print(f"🔧 LLM requested {len(response.tool_calls)} tool call(s)")
        tool_names = [tc['name'] for tc in response.tool_calls]
        print(f"   Tools: {', '.join(tool_names)}")
        
        # Return ALL messages: SystemMessage, HumanMessage (if added), and AIMessage response
        # This ensures the full conversation history is maintained in state
        return {
            "messages": messages_to_add + [response],
            "tools_used": tool_names
        }
    else:
        # Direct response without tools
        print(f"💬 Direct response (no tools needed)")
        print(f"🤖 AI: {response.content}")
        
        # Return ALL messages: SystemMessage, HumanMessage (if added), and AIMessage response
        return {
            "messages": messages_to_add + [response]
        }


def should_continue(state: ToolEnabledState) -> Literal["tools", "end"]:
    """
    Determine if we should execute tools or end the conversation
    
    This is a conditional edge that routes the graph flow
    """
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    # If last message has tool calls, route to tools node
    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, we're done
    return "end"


# ============================================================================
# CONCEPT 4: Building the Graph with Tools
# ============================================================================


def create_tool_enabled_graph():
    """
    Create a graph that supports tool calling:
    
    START -> Chatbot -> [Decision] -> Tools -> Chatbot -> END
                          |
                          └-> END (if no tools needed)
    """
    # Initialize graph
    graph = StateGraph[ToolEnabledState, None, ToolEnabledState, ToolEnabledState](ToolEnabledState)
    
    # Create tools list
    tools = [search_faq, check_order_status, create_support_ticket]
    
    # Add nodes
    graph.add_node("chatbot", chatbot_with_tools)
    graph.add_node("tools", ToolNode(tools))  # ToolNode handles tool execution
    
    # Define flow
    graph.add_edge(START, "chatbot")
    
    # Conditional edge: decide if we need to call tools
    graph.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",  # If tools needed, go to tools node
            "end": END         # Otherwise, end
        }
    )
    
    # After tools execute, go back to chatbot to process results
    graph.add_edge("tools", "chatbot")
    
    return graph.compile()


# ============================================================================
# CONCEPT 5: Running Tool-Enabled Chatbot
# ============================================================================


def run_tool_examples():
    """
    Run examples demonstrating tool usage
    """
    app = create_tool_enabled_graph()
    
    # Example 1: FAQ Search
    print("\n" + "="*60)
    print("📚 EXAMPLE 1: FAQ Search Tool")
    print("="*60)
    
    state1 = {
        "user_input": "What's your return policy?",
        "messages": [],
        "tools_used": []
    }
    
    result1 = app.invoke(state1)
    print(f"\n✅ Tools used: {result1.get('tools_used', [])}")
    print(f"📝 Final response available in messages")
    
    # Example 2: Order Status
    print("\n" + "="*60)
    print("📦 EXAMPLE 2: Order Status Tool")
    print("="*60)
    
    state2 = {
        "user_input": "Can you check the status of order #12345?",
        "messages": [],
        "tools_used": []
    }
    
    result2 = app.invoke(state2)
    print(f"\n✅ Tools used: {result2.get('tools_used', [])}")
    
    # Example 3: No Tool Needed
    print("\n" + "="*60)
    print("💬 EXAMPLE 3: Direct Response (No Tools)")
    print("="*60)
    
    state3 = {
        "user_input": "Thank you for your help!",
        "messages": [],
        "tools_used": []
    }
    
    result3 = app.invoke(state3)
    print(f"\n✅ Tools used: {result3.get('tools_used', [])}")
    
    # Example 4: Creating Support Ticket
    print("\n" + "="*60)
    print("🎫 EXAMPLE 4: Create Support Ticket")
    print("="*60)
    
    state4 = {
        "user_input": "I received a damaged product and need help with a replacement",
        "messages": [],
        "tools_used": []
    }
    
    result4 = app.invoke(state4)
    print(f"\n✅ Tools used: {result4.get('tools_used', [])}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found")
        exit(1)
    
    # Run examples
    run_tool_examples()
    
    print("\n" + "="*60)
    print("💡 KEY TAKEAWAYS")
    print("="*60)
    print("1. Tools are Python functions decorated with @tool")
    print("2. bind_tools() connects tools to the LLM")
    print("3. LLM decides when and which tools to call")
    print("4. ToolNode executes tool calls automatically")
    print("5. Conditional edges route based on tool calls")
    print("6. Graph loops back after tool execution")
    print("="*60)
    
    print("\n🎓 Next Steps:")
    print("   Run 004-conditional-routing.py to learn about")
    print("   intelligent routing based on conversation intent!\n")


# ============================================================================
# DETAILED EXPLANATION: HOW IT WORKS STEP BY STEP
# ============================================================================
"""
HOW THE TOOL-ENABLED CHATBOT WORKS - STEP BY STEP EXPLANATION
================================================================

This script builds a customer support chatbot that can call tools (functions)
to retrieve information or perform actions. The LLM intelligently decides when
to use tools based on the conversation context.

ARCHITECTURE OVERVIEW
---------------------

1. TOOLS (Lines 40-248)
   - Python functions decorated with @tool that LLMs can call
   - Three tools: search_faq, check_order_status, create_support_ticket
   - Each tool has a docstring that describes when and how to use it

2. STATE SCHEMA (Lines 256-267)
   - ToolEnabledState tracks conversation history
   - messages: List of all messages (SystemMessage, HumanMessage, AIMessage, ToolMessage)
   - user_input: Current user query
   - tools_used: List of tools that were called

3. GRAPH STRUCTURE (Lines 410-444)
   - Flow: START -> Chatbot -> [Decision] -> Tools -> Chatbot -> END
   - Conditional edge routes based on whether tools are needed
   - Graph loops back after tool execution

STEP-BY-STEP EXAMPLE WALKTHROUGH
---------------------------------

Example: User asks "What's your return policy?"

STEP 1: INITIAL STATE
   state = {
       "user_input": "What's your return policy?",
       "messages": [],
       "tools_used": []
   }

STEP 2: GRAPH EXECUTION BEGINS
   START → chatbot node

STEP 3: CHATBOT NODE PROCESSES (Lines 296-385)
   a) Messages is empty → adds SystemMessage with instructions
   b) Adds HumanMessage: "What's your return policy?"
   c) Calls LLM with tools bound (create_llm_with_tools)
   d) LLM analyzes the query and decides it needs FAQ information
   e) LLM returns AIMessage with tool_calls requesting search_faq

   State after Step 3:
   {
       "messages": [
           SystemMessage("You are a helpful customer support agent..."),
           HumanMessage("What's your return policy?"),
           AIMessage(tool_calls=[{"name": "search_faq", "args": {"query": "return policy"}}])
       ],
       "tools_used": ["search_faq"]
   }

STEP 4: CONDITIONAL ROUTING (Lines 388-402)
   should_continue() checks last message:
   - Last message has tool_calls → route to "tools"

STEP 5: TOOLS NODE EXECUTES
   a) ToolNode extracts tool calls from AIMessage
   b) Calls search_faq("return policy")
   c) Tool searches FAQ database and finds return policy
   d) Creates ToolMessage with FAQ result

   State after Step 5:
   {
       "messages": [
           SystemMessage(...),
           HumanMessage("What's your return policy?"),
           AIMessage(tool_calls=[...]),
           ToolMessage(content="RETURN POLICY:\n- 30-day return window...")
       ]
   }

STEP 6: BACK TO CHATBOT NODE
   Tools → chatbot (loop back)
   a) Chatbot detects last message is ToolMessage
   b) Doesn't add new user input (already in history)
   c) Calls LLM with full conversation history
   d) LLM sees tool result and generates natural language response

   State after Step 6:
   {
       "messages": [
           SystemMessage(...),
           HumanMessage("What's your return policy?"),
           AIMessage(tool_calls=[...]),
           ToolMessage(...),
           AIMessage(content="Based on our FAQ, here's our return policy...")
       ]
   }

STEP 7: FINAL ROUTING
   should_continue() checks:
   - Last message is AIMessage without tool_calls → route to "end"

STEP 8: END
   Graph execution completes. Final response available in last AIMessage.

VISUAL FLOW DIAGRAM
-------------------

   START
     ↓
   [Chatbot Node]
     ↓
   LLM decides: Need search_faq tool?
     ↓ YES
   [Tools Node]
     ↓
   search_faq("return policy") executed
     ↓
   ToolMessage created with FAQ result
     ↓
   [Chatbot Node] (loops back)
     ↓
   LLM sees tool result, generates final answer
     ↓
   No more tool calls needed
     ↓
   END

KEY CONCEPTS
------------

1. TOOL DECISION: LLM decides when/which tools to use - code doesn't force it
2. GRAPH LOOPING: Chatbot → Tools → Chatbot loops until no more tools needed
3. MESSAGE HISTORY: Complete conversation includes SystemMessage, HumanMessage,
   AIMessage (with tool_calls), and ToolMessage (with results)
4. TOOL EXECUTION: ToolNode automatically executes tool calls and creates ToolMessages
5. CONDITIONAL ROUTING: should_continue() routes based on tool_calls presence

This pattern enables AI agents that can retrieve information and perform actions
based on conversation context, making chatbots much more capable and useful.


# ============================================================================
# FAQ: COMMON QUESTIONS ABOUT THIS CODE
# ============================================================================

FREQUENTLY ASKED QUESTIONS
==========================

Q1: ARE DETAILED COMMENTS REQUIRED FOR EACH TOOL?
-------------------------------------------------
YES! Detailed docstrings are CRITICAL for tools because:

1. LLM Decision Making:
   - The LLM reads the docstring to understand what each tool does
   - It uses the docstring to decide WHEN to call the tool
   - Poor docstrings = LLM won't use tools correctly

2. Tool Schema Generation:
   - The @tool decorator converts the docstring into a tool schema
   - This schema is sent to the LLM so it knows what tools are available

3. Best Practices for Tool Docstrings:
   ✓ Clear description of what the tool does
   ✓ List of scenarios when to use it (like "Use this tool when customers ask about:")
   ✓ Parameter descriptions (Args section)
   ✓ Return value description (Returns section)
   ✓ Example use cases help the LLM understand context

Example of a good tool docstring:
    @tool
    def search_faq(query: str) -> str:
        \"\"\"
        Search the FAQ database for answers to common questions.
        
        Use this tool when customers ask about:
        - Shipping policies
        - Return policies
        - Payment methods
        
        Args:
            query: The customer's question or search query
            
        Returns:
            The relevant FAQ answer if found, or a message indicating no match
        \"\"\"

Without good docstrings, the LLM might:
- Never call the tool when it should
- Call the tool incorrectly
- Use wrong parameters
- Not understand the tool's purpose


Q2: EXPLAIN isinstance() METHOD
--------------------------------
isinstance() is a Python built-in function that checks if an object is an instance
of a specific class (or one of its subclasses).

Syntax:
    isinstance(object, class_or_tuple) -> bool

Returns:
    True if object is an instance of the class, False otherwise

Example Usage:
    # Check if message is a ToolMessage
    last_message = messages[-1]
    if isinstance(last_message, ToolMessage):
        print("This is a ToolMessage!")
    
    # Check if message is ANY type of message
    if isinstance(last_message, (HumanMessage, AIMessage, ToolMessage)):
        print("This is a message object")

In This Code (Line 315):
    is_responding_to_tools = isinstance(last_message, ToolMessage)
    
    This checks if the last message in the conversation is a ToolMessage.
    If True, it means we're responding to tool execution results.
    If False, it means we're processing a new user input.

Why Use isinstance() Instead of type()?
    - isinstance() checks inheritance (subclasses return True)
    - type() only checks exact match
    - isinstance() is more flexible and Pythonic
    
    Example:
        class ParentMessage: pass
        class ChildMessage(ParentMessage): pass
        
        msg = ChildMessage()
        isinstance(msg, ParentMessage)  # True (ChildMessage inherits from ParentMessage)
        type(msg) == ParentMessage      # False (exact type is ChildMessage)


Q3: WHAT IS THE NAME OF THIS PATTERN?
--------------------------------------
This pattern has several names depending on context:

PRIMARY NAMES:
1. "Tool Calling Pattern" or "Function Calling Pattern"
   - Most common in LangChain/LangGraph documentation
   - Describes LLMs calling external functions/tools

2. "Agent Pattern" or "AI Agent Pattern"
   - Describes an autonomous agent that uses tools
   - Common in AI/ML research

3. "ReAct Pattern" (Reasoning + Acting)
   - Reasoning: LLM decides what tool to use
   - Acting: Tool executes the action
   - Observation: Tool result is observed
   - Loop continues until task complete

ALSO KNOWN AS:
- Tool-Augmented LLM: LLM enhanced with external tools
- Tool Use Pattern: Pattern of using tools to extend capabilities
- Orchestration Pattern: Orchestrating LLM and tool execution

GRAPH STRUCTURE PATTERNS:
- State Machine Pattern: Different states (chatbot, tools) with transitions
- Workflow Pattern: Sequential workflow with conditional routing
- Loop Pattern: Graph loops back after tool execution

MOST ACCURATE DESCRIPTION:
"Tool-Calling Agent Pattern" or "Function-Calling Agent Pattern"
- An AI agent that can call tools/functions
- Uses conditional routing to execute tools
- Loops back to process tool results
- Makes decisions based on conversation context

This pattern is fundamental to building:
- AI assistants (like ChatGPT plugins)
- Autonomous agents (like AutoGPT)
- RAG systems (Retrieval-Augmented Generation)
- Tool-augmented chatbots


# ============================================================================
# UNDERSTANDING NODES, TOOLS, AND EDGES IN LANGGRAPH
# ============================================================================

NODES, TOOLS, AND EDGES - FUNDAMENTAL CONCEPTS
===============================================

In LangGraph, these three concepts work together to create intelligent workflows:

1. NODES
--------
A NODE is a processing unit that performs a specific task. It's a function that:
- Receives state as input
- Performs some computation/action
- Returns updated state

Think of nodes as "workstations" in your workflow.

Example from this code (Lines 296-385):
    def chatbot_with_tools(state: ToolEnabledState) -> dict:
        # This is a NODE function
        # It receives state, processes it, returns updated state
        messages = state.get("messages", [])
        # ... processing logic ...
        return {"messages": updated_messages}

Nodes are added to the graph with:
    graph.add_node("chatbot", chatbot_with_tools)
    #              ^name      ^function

In your graph, you have TWO nodes:
1. "chatbot" node - Calls LLM and decides if tools are needed
2. "tools" node - Executes tool calls (uses ToolNode helper)

Visual representation:
    [chatbot node]  ← This processes user input, calls LLM
    [tools node]    ← This executes tool functions


2. TOOLS
--------
A TOOL is a function that the LLM can call to perform actions or retrieve information.
Tools are NOT nodes - they're functions that nodes can execute.

Tools are:
- Python functions decorated with @tool
- Described by docstrings (LLM reads these!)
- Called by the LLM when needed
- Executed by the "tools" node

Example from this code (Lines 40-140):
    @tool
    def search_faq(query: str) -> str:
        # This is a TOOL function
        # LLM can call this to search FAQs
        faq_database = {...}
        return answer

In your code, you have THREE tools:
1. search_faq - Searches FAQ database
2. check_order_status - Checks order information
3. create_support_ticket - Creates support tickets

Tools are bound to the LLM:
    tools = [search_faq, check_order_status, create_support_ticket]
    llm_with_tools = llm.bind_tools(tools)
    # LLM now knows about these tools and can call them

Tools are executed by the "tools" node:
    graph.add_node("tools", ToolNode(tools))
    # ToolNode automatically executes tool calls


3. EDGES
--------
An EDGE defines the flow between nodes - it's the "arrow" connecting nodes.

There are TWO types of edges:

A) REGULAR EDGE (Simple connection)
   Connects one node directly to another, always follows this path.
   
   Example from this code (Lines 429, 442):
       graph.add_edge(START, "chatbot")      # START → chatbot
       graph.add_edge("tools", "chatbot")     # tools → chatbot
   
   Visual:
       START ──→ [chatbot]
       [tools] ──→ [chatbot]

B) CONDITIONAL EDGE (Decision point)
   Routes to different nodes based on state/conditions.
   Uses a routing function to decide which path to take.
   
   Example from this code (Lines 432-439):
       graph.add_conditional_edges(
           "chatbot",           # From this node
           should_continue,     # Use this function to decide
           {
               "tools": "tools",  # If function returns "tools", go to tools node
               "end": END         # If function returns "end", end the graph
           }
       )
   
   The routing function (Lines 388-402):
       def should_continue(state) -> Literal["tools", "end"]:
           # Checks if tools are needed
           if last_message has tool_calls:
               return "tools"  # Route to tools node
           return "end"        # Route to END
   
   Visual:
       [chatbot] ──→ [Decision: should_continue?]
                        ├─ "tools" → [tools node]
                        └─ "end" → END


COMPLETE FLOW VISUALIZATION
----------------------------

Your graph structure:

    START
      │
      │ (regular edge)
      ↓
    [chatbot node]
      │
      │ (conditional edge: should_continue)
      ├─→ "tools" ──→ [tools node] ──→ (regular edge) ──→ [chatbot node]
      │                                                      │
      └─→ "end" ──→ END                                      │
                                                             │
                                                             └─→ (loops back until "end")


KEY DIFFERENCES SUMMARY
------------------------

NODES:
- Processing units/workstations
- Functions that receive and return state
- Examples: chatbot_with_tools(), ToolNode()
- Added with: graph.add_node(name, function)

TOOLS:
- Functions the LLM can call
- Decorated with @tool
- Executed BY nodes (specifically the tools node)
- Examples: search_faq(), check_order_status()
- Bound to LLM with: llm.bind_tools([tools])

EDGES:
- Connections between nodes
- Define workflow direction
- Two types: regular (always) and conditional (if/then)
- Added with: graph.add_edge() or graph.add_conditional_edges()

RELATIONSHIP:
- Nodes USE tools (tools node executes tool functions)
- Edges CONNECT nodes (define the flow)
- Tools are CALLED by LLM (via nodes)

ANALOGY:
Think of a restaurant:
- NODES = Kitchen stations (prep station, cooking station, plating station)
- TOOLS = Kitchen equipment (oven, blender, knife) - used by stations
- EDGES = Workflow path (prep → cook → plate → serve)
"""

