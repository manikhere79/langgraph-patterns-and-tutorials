"""
005 - Multi-Tool Integration with Routing

This tutorial combines conditional routing with multiple tools.
We'll build a sophisticated customer support system that routes queries
and uses appropriate tools based on the conversation context.

Key Concepts:
- Combining routing and tools
- Tool selection based on intent
- Parallel tool execution
- Tool result processing

Learning Objectives:
1. Integrate routing with tool calling
2. Use different tools for different intents
3. Handle multiple tool calls in sequence
4. Process and present tool results effectively
"""

import os
from typing import TypedDict, Annotated, Literal
from operator import add
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 
#from langchain_openai import ChatOpenAI, AzureChatOpenAI   # Commented out - using Gemini instead
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# ============================================================================
# TOOLS DEFINITION
# ============================================================================

@tool
def search_knowledge_base(query: str, category: str = "general") -> str:
    """
    Search the knowledge base for information.
    
    Args:
        query: Search query
        category: Category to search in (billing, technical, shipping, general)
        
    Returns:
        Relevant information from knowledge base
    """
    print(f"\n{'='*60}")
    print(f"🔧 START TOOL EXECUTION: search_knowledge_base")
    print(f"   Parameters: query='{query}', category='{category}'")
    print(f"{'='*60}")
    
    kb = {
        "billing": {
            "refund": "Refunds are processed within 5-7 business days. Contact billing@techshop.com",
            "payment": "We accept all major credit cards, PayPal, and wire transfer for enterprise",
            "invoice": "Invoices are sent within 24 hours of order. Check your email or account portal"
        },
        "technical": {
            "setup": "1. Unbox device 2. Charge fully 3. Download companion app 4. Follow setup wizard",
            "troubleshoot": "Try: restart device, check connections, update firmware, reset to factory settings",
            "compatibility": "Check product specs page for full compatibility matrix"
        },
        "shipping": {
            "tracking": "Enter order number at techshop.com/track or check shipping email",
            "delays": "Current shipping delays: 1-2 days due to high volume. Express unaffected",
            "international": "International shipping: 10-15 business days, customs may add time"
        },
        "general": {
            "help": "For assistance, contact support@techshop.com or call 1-800-TECHSHOP",
            "hours": "Support hours: Monday-Friday 9AM-6PM EST, Saturday 10AM-4PM EST",
            "account": "Manage your account at techshop.com/account or through our mobile app"
        }
    }
    
    category_data = kb.get(category, kb.get("general", {}))
    
    for key, value in category_data.items():
        if key in query.lower():
            print(f"   📚 Found KB article: {category}/{key}")
            return value
    
    return f"No specific article found. Searched in {category} knowledge base."


@tool
def check_order_details(order_id: str) -> str:
    """
    Get detailed order information including items, status, and tracking.
    
    Args:
        order_id: Order number (e.g., ORD-12345)
        
    Returns:
        Order details
    """
    print(f"\n{'='*60}")
    print(f"🔧 START TOOL EXECUTION: check_order_details")
    print(f"   Parameters: order_id='{order_id}'")
    print(f"{'='*60}")
    
    orders = {
        "ORD-12345": {
            "items": ["Laptop Pro 15", "Wireless Mouse"],
            "total": "$1,299.99",
            "status": "Shipped",
            "tracking": "1Z999AA10123456784",
            "eta": "Jan 20, 2024"
        },
        "ORD-12346": {
            "items": ["Keyboard Mechanical"],
            "total": "$149.99",
            "status": "Processing",
            "tracking": None,
            "eta": "Jan 22, 2024"
        }
    }
    
    order = orders.get(order_id)
    if not order:
        return f"Order {order_id} not found. Please verify the order number."
    
    print(f"   📦 Retrieved order: {order_id}")
    
    result = f"""
    Order: {order_id}
    Items: {', '.join(order['items'])}
    Total: {order['total']}
    Status: {order['status']}
    Expected: {order['eta']}
    """
    
    if order['tracking']:
        result += f"\n    Tracking: {order['tracking']}"
    
    return result


@tool
def check_inventory(product_name: str) -> str:
    """
    Check product availability and stock levels.
    
    Args:
        product_name: Name of the product
        
    Returns:
        Inventory status and availability
    """
    print(f"\n{'='*60}")
    print(f"🔧 START TOOL EXECUTION: check_inventory")
    print(f"   Parameters: product_name='{product_name}'")
    print(f"{'='*60}")
    
    inventory = {
        "laptop pro 15": {"stock": 45, "warehouse": "NY", "next_shipment": "Jan 25"},
        "wireless mouse": {"stock": 230, "warehouse": "CA", "next_shipment": None},
        "keyboard mechanical": {"stock": 0, "warehouse": "TX", "next_shipment": "Feb 1"},
        "monitor 4k": {"stock": 12, "warehouse": "NY", "next_shipment": "Jan 30"}
    }
    
    product_key = product_name.lower()
    if product_key in inventory:
        item = inventory[product_key]
        print(f"   📊 Checked inventory: {product_name}")
        
        if item["stock"] > 0:
            return f"{product_name}: {item['stock']} units in stock at {item['warehouse']} warehouse. Order now for immediate shipping!"
        else:
            return f"{product_name}: Currently out of stock. Next shipment expected {item['next_shipment']}. Pre-order available!"
    
    return f"Product '{product_name}' not found in inventory system."


@tool
def create_ticket(title: str, category: str, priority: str = "medium") -> str:
    """
    Create a support ticket for issues requiring human attention.
    
    Args:
        title: Brief title of the issue
        category: Category (billing, technical, shipping)
        priority: Priority level (low, medium, high, urgent)
        
    Returns:
        Ticket confirmation
    """
    print(f"\n{'='*60}")
    print(f"🔧 START TOOL EXECUTION: create_ticket")
    print(f"   Parameters: title='{title}', category='{category}', priority='{priority}'")
    print(f"{'='*60}")
    
    import random
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    
    print(f"   🎫 Created ticket: {ticket_id}")
    
    eta = {
        "low": "48 hours",
        "medium": "24 hours",
        "high": "4 hours",
        "urgent": "1 hour"
    }
    
    return f"""
    Ticket Created: {ticket_id}
    Category: {category}
    Priority: {priority.upper()}
    
    A specialist will respond within {eta.get(priority, '24 hours')}.
    You'll receive updates via email.
    """


@tool
def process_refund(order_id: str, reason: str) -> str:
    """
    Initiate refund process for an order.
    
    Args:
        order_id: Order number to refund
        reason: Reason for refund
        
    Returns:
        Refund confirmation
    """
    print(f"\n{'='*60}")
    print(f"🔧 START TOOL EXECUTION: process_refund")
    print(f"   Parameters: order_id='{order_id}', reason='{reason}'")
    print(f"{'='*60}")
    print(f"   💰 Processing refund for: {order_id}")
    
    return f"""
    Refund Initiated for {order_id}
    
    Reason: {reason}
    Refund Amount: Will match original payment
    Processing Time: 5-7 business days
    Method: Original payment method
    
    You'll receive email confirmation shortly.
    Reference: REF-{hash(order_id) % 10000}
    """


# ============================================================================
# STATE DEFINITION
# ============================================================================


class MultiToolState(TypedDict):
    """
    State for multi-tool support system
    """
    # User input
    user_input: str
    
    # Classified intent
    intent: str
    
    # Conversation messages
    messages: Annotated[list, add]
    
    # Tools used in this conversation
    tools_used: Annotated[list[str], add]
    
    # Final response
    response: str


# ============================================================================
# STATE PRINTING UTILITY
# ============================================================================


def print_state_properties(state: MultiToolState, node_name: str = "") -> None:
    """
    Print all properties of the current state in a formatted way.
    
    Args:
        state: The current state dictionary
        node_name: Optional name of the node/function calling this
    """
    print(f"\n{'='*60}")
    if node_name:
        print(f"📊 STATE PROPERTIES - {node_name}")
    else:
        print("📊 STATE PROPERTIES")
    print(f"{'='*60}")
    
    # Print each state property
    print(f"👤 user_input: {state.get('user_input', 'N/A')}")
    print(f"🎯 intent: {state.get('intent', 'N/A')}")
    
    # Iterate and print all messages
    messages = state.get('messages', [])
    print(f"💬 messages: {len(messages)} message(s)")
    if messages:
        for idx, msg in enumerate(messages, 1):
            msg_type = type(msg).__name__
            if hasattr(msg, 'content'):
                content = str(msg.content)
                # Truncate very long content
                if len(content) > 200:
                    content_preview = content[:200] + "..."
                else:
                    content_preview = content
                print(f"   [{idx}] {msg_type}: {content_preview}")
            else:
                print(f"   [{idx}] {msg_type}: (no content)")
            
            # Print tool calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"       └─ Tool calls: {len(msg.tool_calls)}")
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    print(f"          • {tool_name}({', '.join(f'{k}={v}' for k, v in tool_args.items())})")
    else:
        print(f"   (no messages)")
    
    # Print tools used
    tools_used = state.get('tools_used', [])
    print(f"🔧 tools_used: {len(tools_used)} tool(s)")
    if tools_used:
        for idx, tool in enumerate(tools_used, 1):
            print(f"   [{idx}] {tool}")
    
    # Print response
    response = state.get('response', '')
    if response:
        response_preview = response[:100] + "..." if len(response) > 100 else response
        print(f"💬 response: {response_preview}")
    else:
        print(f"💬 response: (empty)")
    
    print(f"{'='*60}\n")


# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================


def classify_intent_node(state: MultiToolState) -> dict:
    """
    Classify user intent for routing
    """
    print_state_properties(state, "CLASSIFY INTENT")
    print(f"\n{'='*60}")
    print("🎯 CLASSIFYING INTENT")
    print(f"{'='*60}")
    
    user_input = state["user_input"]
    print(f"👤 Input: {user_input}")
    
    # Simple keyword-based classification for demo
    # In production, use LLM or ML model
    intent = "general"
    
    if any(word in user_input.lower() for word in ["refund", "charge", "payment", "bill", "invoice"]):
        intent = "billing"
    elif any(word in user_input.lower() for word in ["not working", "broken", "error", "technical", "setup"]):
        intent = "technical"
    elif any(word in user_input.lower() for word in ["order", "shipping", "delivery", "track", "package"]):
        intent = "shipping"
    elif any(word in user_input.lower() for word in ["stock", "available", "inventory", "buy"]):
        intent = "inventory"
    
    print(f"📊 Intent: {intent}")
    
    return {
        "intent": intent,
        "messages": [HumanMessage(content=user_input)]
    }


# ============================================================================
# TOOL-ENABLED HANDLER NODES
# ============================================================================


def create_specialized_llm(intent: str):
    """
    Create LLM with tools appropriate for the intent
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )
    
    # # Azure OpenAI configuration
    # # Using AzureChatOpenAI for proper Azure OpenAI integration
    # azure_endpoint = os.getenv("GENAI_AZURE_OPENAI_ENDPOINT")
    # api_key = os.getenv("GENAI_AZURE_OPENAI_API_KEY")
    # api_version = os.getenv("GENAI_AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    # deployment_name = os.getenv("GENAI_AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    
    # # Validate required environment variables
    # if not api_key:
    #     raise ValueError("GENAI_AZURE_OPENAI_API_KEY environment variable is required")
    # if not azure_endpoint:
    #     raise ValueError("GENAI_AZURE_OPENAI_ENDPOINT environment variable is required")
    
    # # Ensure API key is a string (not None)
    # if not isinstance(api_key, str) or api_key.strip() == "":
    #     raise ValueError("GENAI_AZURE_OPENAI_API_KEY must be a non-empty string")
    
    # # Use AzureChatOpenAI for Azure OpenAI (recommended approach)
    # llm = AzureChatOpenAI(
    #     azure_endpoint=azure_endpoint,
    #     api_key=api_key,
    #     api_version=api_version,
    #     deployment_name=deployment_name,
    #     temperature=0.7,
    # )
    
    # Assign tools based on intent
    tool_map = {
        "billing": [search_knowledge_base, check_order_details, process_refund, create_ticket],
        "technical": [search_knowledge_base, check_order_details, create_ticket],
        "shipping": [check_order_details, search_knowledge_base, create_ticket],
        "inventory": [check_inventory, search_knowledge_base],
        "general": [search_knowledge_base, create_ticket]
    }
    
    tools = tool_map.get(intent, [search_knowledge_base])
    return llm.bind_tools(tools), tools


def billing_handler(state: MultiToolState) -> dict:
    """
    Handle billing queries with appropriate tools
    """
    print_state_properties(state, "BILLING HANDLER")
    print(f"\n{'='*60}")
    print("💳 BILLING HANDLER WITH TOOLS")
    print(f"{'='*60}")
    
    llm_with_tools, tools = create_specialized_llm("billing")
    
    system_msg = SystemMessage(content="""
You are a billing specialist with access to:
- Knowledge base search for policies
- Order details lookup
- Refund processing
- Ticket creation

Help customers with billing issues professionally and efficiently.
Use tools when you need specific information.
    """)
    
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        print(f"🔧 Calling {len(response.tool_calls)} tool(s)")
        return {"messages": [response]}
    
    print(f"💬 Direct response")
    return {"messages": [response], "response": response.content}


def technical_handler(state: MultiToolState) -> dict:
    """
    Handle technical queries with appropriate tools
    """
    print_state_properties(state, "TECHNICAL HANDLER")
    print(f"\n{'='*60}")
    print("🔧 TECHNICAL HANDLER WITH TOOLS")
    print(f"{'='*60}")
    
    llm_with_tools, tools = create_specialized_llm("technical")
    
    system_msg = SystemMessage(content="""
You are a technical support specialist with access to:
- Technical knowledge base
- Order details (to check product)
- Ticket creation for complex issues

Provide step-by-step troubleshooting. Ask clarifying questions.
    """)
    
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        print(f"🔧 Calling {len(response.tool_calls)} tool(s)")
        return {"messages": [response]}
    
    print(f"💬 Direct response")
    return {"messages": [response], "response": response.content}


def shipping_handler(state: MultiToolState) -> dict:
    """
    Handle shipping queries with appropriate tools
    """
    print_state_properties(state, "SHIPPING HANDLER")
    print(f"\n{'='*60}")
    print("📦 SHIPPING HANDLER WITH TOOLS")
    print(f"{'='*60}")
    
    llm_with_tools, tools = create_specialized_llm("shipping")
    
    system_msg = SystemMessage(content="""
You are a shipping specialist with access to:
- Order tracking and details
- Shipping knowledge base
- Ticket creation

Proactively provide tracking information and delivery estimates.
    """)
    
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        print(f"🔧 Calling {len(response.tool_calls)} tool(s)")
        return {"messages": [response]}
    
    print(f"💬 Direct response")
    return {"messages": [response], "response": response.content}


def inventory_handler(state: MultiToolState) -> dict:
    """
    Handle inventory/product availability queries
    """
    print_state_properties(state, "INVENTORY HANDLER")
    print(f"\n{'='*60}")
    print("📊 INVENTORY HANDLER WITH TOOLS")
    print(f"{'='*60}")
    
    llm_with_tools, tools = create_specialized_llm("inventory")
    
    system_msg = SystemMessage(content="""
You are a product specialist with access to:
- Real-time inventory checking
- Product knowledge base

Help customers find product availability and suggest alternatives if out of stock.
    """)
    
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        print(f"🔧 Calling {len(response.tool_calls)} tool(s)")
        return {"messages": [response]}
    
    print(f"💬 Direct response")
    return {"messages": [response], "response": response.content}


# ============================================================================
# ROUTING AND TOOL EXECUTION
# ============================================================================


def route_to_handler(state: MultiToolState) -> str:
    """
    Route to appropriate handler based on intent
    """
    print_state_properties(state, "ROUTE TO HANDLER")
    intent = state.get("intent", "general")
    print(f"\n🔀 Routing to: {intent}")
    return intent


def should_continue_tools(state: MultiToolState) -> Literal["tools", "end"]:
    """
    Check if we need to execute tools
    """
    print_state_properties(state, "SHOULD CONTINUE TOOLS")
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    return "end"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_multi_tool_graph():
    """
    Create graph with routing and multiple tools
    
    START -> Classify -> Route -> Handler -> [Tools?] -> END
                           |                     ↓
                           |                   Tools
                           |                     ↓
                           |                  Handler
    """
    graph = StateGraph(MultiToolState)
    
    # Add nodes
    graph.add_node("classify", classify_intent_node)
    graph.add_node("billing", billing_handler)
    graph.add_node("technical", technical_handler)
    graph.add_node("shipping", shipping_handler)
    graph.add_node("inventory", inventory_handler)
    
    # Create comprehensive tool list
    all_tools = [
        search_knowledge_base,
        check_order_details,
        check_inventory,
        create_ticket,
        process_refund
    ]
    graph.add_node("tools", ToolNode(all_tools))
    
    # Flow
    graph.add_edge(START, "classify")
    
    # Route from classify to handlers
    graph.add_conditional_edges(
        "classify",
        route_to_handler,
        {
            "billing": "billing",
            "technical": "technical",
            "shipping": "shipping",
            "inventory": "inventory",
            "general": "billing"  # Default to billing as general handler
        }
    )
    
    # Each handler checks if tools needed
    for handler in ["billing", "technical", "shipping", "inventory"]:
        graph.add_conditional_edges(
            handler,
            should_continue_tools,
            {
                "tools": "tools",
                "end": END
            }
        )
    
    # After tools, route back based on original intent
    graph.add_conditional_edges(
        "tools",
        lambda state: state.get("intent", "billing"),
        {
            "billing": "billing",
            "technical": "technical",
            "shipping": "shipping",
            "inventory": "inventory",
            "general": "billing"
        }
    )
    
    return graph.compile()


# ============================================================================
# EXAMPLES
# ============================================================================


def run_examples():
    """
    Run comprehensive examples
    """
    app = create_multi_tool_graph()
    
    test_cases = [
        "I want a refund for order ORD-12345",
        "Is the Laptop Pro 15 in stock?",
        "Where is my order ORD-12346?",
        "My device is not turning on, I need help",
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'#'*60}")
        print(f"TEST CASE {i}")
        print(f"{'#'*60}")
        
        state = {
            "user_input": query,
            "intent": "",
            "messages": [],
            "tools_used": [],
            "response": ""
        }
        
        result = app.invoke(state)
        
        print(f"\n{'─'*60}")
        print("SUMMARY")
        print(f"{'─'*60}")
        print(f"Intent: {result['intent']}")
        print(f"Tools Used: {result.get('tools_used', [])}")



def run_examples1():
    app = create_multi_tool_graph()
    query = "Hi, I want a refund for order ORD-12345. I was charged twice!"
    state = {
            "user_input": query,
            "intent": "",
            "messages": [],
            "tools_used": [],
            "response": ""
        }
        
    result = app.invoke(state)

    print(f"\n{'─'*60}")
    print("SUMMARY")
    print(f"{'─'*60}")
    print(f"Intent: {result['intent']}")
    print(f"Tools Used: {result.get('tools_used', [])}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Check API key - Updated for Azure OpenAI
    if not os.getenv("GENAI_AZURE_OPENAI_API_KEY"):
        print("❌ Error: GENAI_AZURE_OPENAI_API_KEY not found")
        print("   Please set the following in your .env file:")
        print("   - GENAI_AZURE_OPENAI_API_KEY")
        print("   - GENAI_AZURE_OPENAI_ENDPOINT")
        print("   - GENAI_AZURE_OPENAI_DEPLOYMENT_NAME (optional, defaults to 'gpt-4')")
        print("   - GENAI_AZURE_OPENAI_API_VERSION (optional, defaults to '2024-02-15-preview')")
        exit(1)
    
    run_examples1()
    
    print("\n" + "="*60)
    print("💡 KEY TAKEAWAYS")
    print("="*60)
    print("1. Different intents can use different tool sets")
    print("2. Tools are assigned based on handler specialization")
    print("3. Routing + Tools = Powerful agent system")
    print("4. ToolNode handles execution automatically")
    print("5. Graph loops allow tool results to be processed")
    print("="*60)
    
    print("\n🎓 Next Steps:")
    print("   Run 006-memory-checkpointing.py to learn about")
    print("   conversation persistence and memory!\n")


# ============================================================================
# COMPREHENSIVE EXPLANATION: MULTI-TOOL INTEGRATION WITH ROUTING
# ============================================================================
"""
HOW MULTI-TOOL INTEGRATION WORKS - REAL-WORLD EXAMPLE
======================================================

This lesson combines the BEST of both worlds:
- Conditional Routing (Lesson 004): Route to specialized handlers
- Tool Calling (Lesson 003): Use tools for specific actions

Think of it like a hospital: Patients are routed to specialists (routing),
and each specialist has access to specific medical equipment (tools).

REAL-WORLD SCENARIO: Customer Support with Specialized Tools
-------------------------------------------------------------

Scenario: Maria contacts TechShop support about a refund request.

STEP 1: CUSTOMER REACHES OUT
   Maria: "Hi, I want a refund for order ORD-12345. I was charged twice!"

STEP 2: INTENT CLASSIFICATION (classify_intent_node)
   The system analyzes Maria's message:
   - Detects keywords: "refund", "charge"
   - Classifies intent: "billing"
   
   State after classification:
   {
       "user_input": "I want a refund for order ORD-12345...",
       "intent": "billing",
       "messages": [HumanMessage("I want a refund...")]
   }

STEP 3: ROUTING TO SPECIALIST (route_to_handler)
   route_to_handler() checks intent = "billing"
   Returns: "billing"
   Graph routes to: billing_handler node

STEP 4: BILLING HANDLER ACTIVATES WITH SPECIALIZED TOOLS
   The billing handler is like a billing specialist who:
   - Has access to billing-specific tools:
     ✓ search_knowledge_base (for refund policies)
     ✓ check_order_details (to verify the order)
     ✓ process_refund (to actually process refund)
     ✓ create_ticket (if escalation needed)
   
   Note: The billing handler ONLY gets billing-relevant tools!
   It doesn't have inventory tools (that's for inventory handler).
   
   The handler creates a specialized LLM:
   llm_with_tools = llm.bind_tools([
       search_knowledge_base,
       check_order_details,
       process_refund,
       create_ticket
   ])

STEP 5: LLM DECIDES TO USE TOOLS
   The billing specialist LLM analyzes Maria's request:
   - "I need to check the order details first"
   - "Then I'll process the refund"
   - "I should also check refund policy"
   
   LLM decides to call tools:
   AIMessage with tool_calls:
   [
       {"name": "check_order_details", "args": {"order_id": "ORD-12345"}},
       {"name": "search_knowledge_base", "args": {"query": "refund policy", "category": "billing"}}
   ]

STEP 6: TOOL EXECUTION (ToolNode)
   should_continue_tools() detects tool_calls → routes to "tools" node
   
   ToolNode executes:
   1. check_order_details("ORD-12345")
      → Returns: Order details (items, total, status)
   
   2. search_knowledge_base("refund policy", "billing")
      → Returns: "Refunds are processed within 5-7 business days..."
   
   ToolNode creates ToolMessages with results

STEP 7: BACK TO HANDLER (Loop Back)
   After tools execute, graph routes back to billing_handler
   (based on original intent stored in state)
   
   The billing handler LLM now sees:
   - Original user message
   - Tool results (order details + refund policy)
   
   LLM processes results and generates response:
   "I've verified your order ORD-12345. I can process your refund 
   immediately. According to our policy, refunds take 5-7 business 
   days. Should I proceed?"

STEP 8: LLM DECIDES TO PROCESS REFUND
   LLM decides: "Yes, I should process the refund now"
   Calls: process_refund("ORD-12345", "Double charge")

STEP 9: REFUND TOOL EXECUTES
   ToolNode executes process_refund
   → Returns: Refund confirmation with reference number

STEP 10: FINAL RESPONSE
   Handler LLM processes refund confirmation
   → Generates final response:
   "Perfect! I've initiated your refund for order ORD-12345. 
   Reference: REF-1234. The refund will appear in your account 
   within 5-7 business days. You'll receive email confirmation shortly."

STEP 11: COMPLETION
   No more tool calls needed → should_continue_tools() returns "end"
   Graph completes successfully!

VISUAL FLOW DIAGRAM:
--------------------

    START
      │
      ↓
[classify_intent] → Intent: "billing"
      │
      ↓
[route_to_handler] → "billing"
      │
      ↓
[billing_handler] ← Has specialized tools:
      │              - check_order_details
      │              - process_refund
      │              - search_knowledge_base
      │              - create_ticket
      │
      │ LLM decides: Need tools!
      │
      ↓
[should_continue_tools] → "tools"
      │
      ↓
[tools node] ← Executes:
      │         - check_order_details("ORD-12345")
      │         - search_knowledge_base("refund policy")
      │
      │ Creates ToolMessages with results
      │
      ↓
[route back] → Based on intent: "billing"
      │
      ↓
[billing_handler] ← Sees tool results
      │              Processes and decides:
      │              "I'll process the refund"
      │
      ↓
[should_continue_tools] → "tools" (refund tool needed)
      │
      ↓
[tools node] ← Executes:
      │         - process_refund("ORD-12345", "Double charge")
      │
      ↓
[route back] → "billing"
      │
      ↓
[billing_handler] ← Sees refund confirmation
      │              Generates final response
      │
      ↓
[should_continue_tools] → "end" (no more tools)
      │
      ↓
    END


KEY CONCEPTS EXPLAINED
======================

1. SPECIALIZED TOOL SETS PER HANDLER
   ──────────────────────────────────────────────────────────────
   Each handler gets ONLY the tools it needs:
   
   Billing Handler Tools:
   - search_knowledge_base (billing policies)
   - check_order_details (verify orders)
   - process_refund (process refunds)
   - create_ticket (escalate if needed)
   
   Technical Handler Tools:
   - search_knowledge_base (technical docs)
   - check_order_details (check what product they have)
   - create_ticket (for complex issues)
   ❌ NO process_refund (not relevant for technical issues)
   
   Inventory Handler Tools:
   - check_inventory (check stock)
   - search_knowledge_base (product info)
   ❌ NO process_refund (not relevant for inventory)
   
   Why? Security and efficiency:
   - Billing handler shouldn't accidentally process refunds for 
     technical issues
   - Each specialist only sees tools relevant to their domain
   - Reduces errors and improves focus

2. TOOL SELECTION LOGIC (create_specialized_llm)
   ──────────────────────────────────────────────────────────────
   
   def create_specialized_llm(intent: str):
       tool_map = {
           "billing": [search_knowledge_base, check_order_details, 
                      process_refund, create_ticket],
           "technical": [search_knowledge_base, check_order_details, 
                        create_ticket],
           "shipping": [check_order_details, search_knowledge_base, 
                       create_ticket],
           "inventory": [check_inventory, search_knowledge_base]
       }
       tools = tool_map.get(intent, [search_knowledge_base])
       return llm.bind_tools(tools)
   
   This function assigns tools based on intent:
   - Billing intent → Gets refund processing tools
   - Technical intent → Gets troubleshooting tools
   - Inventory intent → Gets stock checking tools
   
   The LLM only sees tools relevant to its specialization!

3. THE LOOP PATTERN (Handler → Tools → Handler)
   ──────────────────────────────────────────────────────────────
   
   This is the KEY pattern:
   
   Handler → Checks if tools needed → Tools → Back to Handler
   
   Why loop back?
   - Handler needs to PROCESS tool results
   - Handler might need MULTIPLE tool calls
   - Handler generates final response after seeing tool results
   
   Example flow:
   1. Handler: "I need order details" → Calls check_order_details
   2. Tools execute → Returns order info
   3. Handler: "Now I'll process refund" → Calls process_refund
   4. Tools execute → Returns refund confirmation
   5. Handler: "Perfect! Here's your confirmation..." → Final response

4. CONDITIONAL TOOL EXECUTION (should_continue_tools)
   ──────────────────────────────────────────────────────────────
   
   def should_continue_tools(state) -> Literal["tools", "end"]:
       last_message = state["messages"][-1]
       
       if last_message has tool_calls:
           return "tools"  # Execute tools
       else:
           return "end"     # No tools needed, done!
   
   This function decides:
   - If LLM requested tools → Route to tools node
   - If LLM gave direct response → End conversation
   
   This allows handlers to:
   - Sometimes use tools (when needed)
   - Sometimes respond directly (when no tools needed)

5. ROUTING BACK AFTER TOOLS
   ──────────────────────────────────────────────────────────────
   
   After tools execute, we need to route back to the CORRECT handler:
   
   graph.add_conditional_edges(
       "tools",
       lambda state: state.get("intent", "billing"),
       {
           "billing": "billing",      # Route back to billing handler
           "technical": "technical",  # Route back to technical handler
           # ... etc
       }
   )
   
   Why? Because:
   - Tools don't know which handler called them
   - We need to return to the SAME handler that requested tools
   - Handler processes tool results and continues conversation


REAL-WORLD ANALOGY: HOSPITAL SYSTEM
===================================

Think of this like a hospital:

1. PATIENT ARRIVES (User query)
   Patient: "I have chest pain"

2. TRIAGE (classify_intent)
   Nurse classifies: "Cardiac issue" → Routes to Cardiology

3. SPECIALIST ASSIGNED (route_to_handler)
   Patient goes to: Cardiology Department

4. CARDIOLOGIST WITH SPECIALIZED EQUIPMENT (Handler with tools)
   Cardiologist has access to:
   - EKG machine (check_order_details)
   - Blood pressure monitor (search_knowledge_base)
   - Defibrillator (process_refund - emergency action)
   - Patient records (create_ticket)
   
   ❌ Cardiologist doesn't have:
   - X-ray machine (that's for Radiology)
   - Lab equipment (that's for Pathology)

5. DIAGNOSIS AND TREATMENT (Tool execution)
   Cardiologist: "I need to run an EKG" → Uses EKG tool
   EKG results come back
   Cardiologist: "I need to check blood pressure" → Uses BP tool
   BP results come back
   Cardiologist: "Based on results, I'll prescribe medication"

6. COMPLETION
   Treatment complete → Patient discharged


COMPARISON: LESSON 003 vs LESSON 004 vs LESSON 005
===================================================

LESSON 003: Tools Only
──────────────────────
- Single LLM with all tools
- LLM decides which tools to use
- Simple: Tools → Process → Done

LESSON 004: Routing Only
────────────────────────
- Route to specialized handlers
- Each handler has its own LLM
- No tools, just specialized responses

LESSON 005: Routing + Tools (BEST!)
───────────────────────────────────
- Route to specialized handlers ✓
- Each handler has specialized tools ✓
- Handlers can use tools as needed ✓
- Combines benefits of both approaches ✓


WHY THIS PATTERN IS POWERFUL
=============================

1. SPECIALIZATION
   Each handler is an expert in its domain with relevant tools
   - Billing expert → Billing tools
   - Technical expert → Technical tools
   - Shipping expert → Shipping tools

2. SECURITY
   Handlers only see tools they need
   - Can't accidentally process refunds for technical issues
   - Reduces risk of misuse

3. EFFICIENCY
   - Route once to specialist
   - Specialist uses tools as needed
   - No need to route multiple times

4. FLEXIBILITY
   - Handlers can use tools OR respond directly
   - Multiple tool calls in sequence
   - Tool results inform next actions

5. SCALABILITY
   - Easy to add new handlers
   - Easy to add new tools
   - Each handler independent


STEP-BY-STEP CODE EXECUTION
===========================

For query: "I want a refund for order ORD-12345"

1. classify_intent_node(state)
   - Analyzes: "refund" → "billing"
   - Returns: {"intent": "billing", "messages": [...]}

2. route_to_handler(state)
   - Gets intent: "billing"
   - Returns: "billing"

3. billing_handler(state)
   - Creates LLM with billing tools
   - LLM analyzes: "Need to check order and process refund"
   - Returns: AIMessage with tool_calls

4. should_continue_tools(state)
   - Checks: Has tool_calls? Yes!
   - Returns: "tools"

5. ToolNode executes
   - Runs: check_order_details("ORD-12345")
   - Creates: ToolMessage with order details

6. Route back to billing_handler
   - Intent still "billing" → Routes to billing_handler

7. billing_handler(state) - Second time
   - Sees tool results
   - LLM: "Now I'll process the refund"
   - Returns: AIMessage with process_refund tool_call

8. ToolNode executes again
   - Runs: process_refund("ORD-12345", "refund request")
   - Creates: ToolMessage with refund confirmation

9. Route back to billing_handler - Third time
   - Processes refund confirmation
   - Generates final response
   - No more tool_calls

10. should_continue_tools(state)
    - Checks: Has tool_calls? No!
    - Returns: "end"

11. END - Conversation complete!


KEY TAKEAWAYS
=============

1. ROUTING + TOOLS = POWERFUL COMBINATION
   - Route to specialists (routing)
   - Specialists use tools (tools)
   - Best of both worlds!

2. SPECIALIZED TOOL SETS
   - Each handler gets only relevant tools
   - Security and efficiency benefits

3. LOOP PATTERN
   - Handler → Tools → Handler (loop)
   - Allows multiple tool calls
   - Handler processes results

4. CONDITIONAL TOOL EXECUTION
   - Handlers can use tools OR respond directly
   - Flexible based on need

5. SCALABLE ARCHITECTURE
   - Easy to add handlers
   - Easy to add tools
   - Each component independent

This pattern is used in production systems for:
- Customer support (like this example)
- Multi-agent systems
- Workflow automation
- Enterprise AI assistants
"""
