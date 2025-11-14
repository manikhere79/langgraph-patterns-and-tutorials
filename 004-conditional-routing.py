"""
004 - Conditional Routing in LangGraph

This tutorial covers CONDITIONAL EDGES - routing graph flow based on state.
We'll build a customer support system that routes queries to specialized handlers.

Key Concepts:
- Conditional edges with routing functions
- Intent classification
- Dynamic graph flow
- Specialized nodes for different query types

Learning Objectives:
1. Implement conditional routing logic
2. Classify user intents
3. Route to specialized handlers
4. Understand graph flow control
"""

import os
from typing import TypedDict, Annotated, Literal
from operator import add
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Import Pydantic models for structured output
# Note: Dynamic import path to access src/models from lesson files
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
from models.structured_output import IntentClassification

# Load environment variables
load_dotenv()

# ============================================================================
# CONCEPT 1: State with Routing Information
# ============================================================================


class RoutingState(TypedDict):
    """
    State that tracks routing and intent classification
    """
    # User input
    user_input: str
    
    # Classified intent (billing, technical, general, etc.)
    intent: str
    
    # Confidence score for intent classification
    confidence: float
    
    # Conversation messages
    messages: Annotated[list, add]
    
    # Response from the system
    response: str
    
    # Routing path taken (for debugging)
    route_taken: str


# ============================================================================
# CONCEPT 2: Intent Classification Node with Structured Output
# ============================================================================
"""
STRUCTURED OUTPUT PATTERN:
Instead of parsing text responses manually (error-prone), we use Pydantic models
to get type-safe, validated structured data from the LLM.

Benefits:
- Type safety: IDE autocomplete and type checking
- Validation: Automatic validation of confidence scores, etc.
- No parsing errors: LLM returns structured JSON, not free-form text
- Better error handling: Clear validation errors instead of parsing failures
"""


def classify_intent(state: RoutingState) -> dict:
    """
    Classify the user's intent using LLM
    
    This node determines what kind of support the user needs:
    - billing: Payment, refunds, charges
    - technical: Product issues, troubleshooting
    - shipping: Delivery, tracking, shipping questions
    - general: FAQs, policies, general info
    - escalation: Complex issues needing human help
    
    Args:
        state: Current state with user input
        
    Returns:
        State updates with intent and confidence
    """
    print(f"\n{'='*60}")
    print("🎯 INTENT CLASSIFICATION NODE")
    print(f"{'='*60}")
    
    user_input = state["user_input"]
    print(f"👤 User: {user_input}")
    
    # Create LLM for classification with structured output
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,  # Low temperature for consistent classification
    )
    
    # Bind structured output model - this ensures type-safe, validated responses
    llm_structured = llm.with_structured_output(IntentClassification)
    
    # Classification prompt - simpler now since we get structured output
    classification_prompt = f"""
Classify the following customer support query into ONE of these categories:

Categories:
1. billing - Questions about payments, refunds, charges, invoices, pricing
2. technical - Product not working, troubleshooting, technical issues, bugs
3. shipping - Delivery status, tracking, shipping costs, shipping address
4. general - Store policies, FAQs, hours, general information
5. escalation - Angry customer, complex issue, requests human support

Query: "{user_input}"

Provide your classification with confidence score and reasoning.
"""
    
    messages = [HumanMessage(content=classification_prompt)]
    
    # Get structured, validated response - no manual parsing needed!
    try:
        classification = llm_structured.invoke(messages)
        
        # Access validated fields directly
        intent = classification.intent
        confidence = classification.confidence
        reason = classification.reason
        
        print(f"📊 Classification Result:")
        print(f"   Intent: {intent}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Reason: {reason}")
        
    except Exception as e:
        print(f"⚠️  Classification error: {e}. Using defaults.")
        # Fallback to defaults if structured output fails
        intent = "general"
        confidence = 0.5
    
    return {
        "intent": intent,
        "confidence": confidence,
        "messages": [HumanMessage(content=user_input)]
    }


# ============================================================================
# CONCEPT 3: Specialized Handler Nodes
# ============================================================================


def handle_billing(state: RoutingState) -> dict:
    """
    Specialized handler for billing-related queries
    """
    print(f"\n{'='*60}")
    print("💳 BILLING HANDLER")
    print(f"{'='*60}")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    system_prompt = SystemMessage(content="""
You are a billing specialist for TechShop. You help customers with:
- Payment issues and methods
- Refund requests and status
- Billing disputes
- Invoice questions
- Pricing inquiries

Be professional, clear about financial policies, and empathetic.
Always verify customer identity before discussing account details.
    """)
    
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    
    print(f"💬 Response: {response.content[:100]}...")
    
    return {
        "messages": [AIMessage(content=response.content)],
        "response": response.content,
        "route_taken": "billing"
    }


def handle_technical(state: RoutingState) -> dict:
    """
    Specialized handler for technical support queries
    """
    print(f"\n{'='*60}")
    print("🔧 TECHNICAL SUPPORT HANDLER")
    print(f"{'='*60}")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    system_prompt = SystemMessage(content="""
You are a technical support specialist for TechShop. You help with:
- Product troubleshooting
- Setup and installation
- Technical specifications
- Compatibility questions
- Error messages and bugs

Provide step-by-step solutions, ask clarifying questions,
and escalate to engineering if needed.
    """)
    
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    
    print(f"💬 Response: {response.content[:100]}...")
    
    return {
        "messages": [AIMessage(content=response.content)],
        "response": response.content,
        "route_taken": "technical"
    }


def handle_shipping(state: RoutingState) -> dict:
    """
    Specialized handler for shipping and delivery queries
    """
    print(f"\n{'='*60}")
    print("📦 SHIPPING HANDLER")
    print(f"{'='*60}")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    system_prompt = SystemMessage(content="""
You are a shipping specialist for TechShop. You help with:
- Order tracking and status
- Delivery estimates
- Shipping costs and options
- Address changes
- Lost or delayed packages

Be proactive in offering tracking information and solutions.
    """)
    
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    
    print(f"💬 Response: {response.content[:100]}...")
    
    return {
        "messages": [AIMessage(content=response.content)],
        "response": response.content,
        "route_taken": "shipping"
    }


def handle_general(state: RoutingState) -> dict:
    """
    Handler for general queries and FAQs
    """
    print(f"\n{'='*60}")
    print("ℹ️  GENERAL SUPPORT HANDLER")
    print(f"{'='*60}")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    system_prompt = SystemMessage(content="""
You are a general customer support agent for TechShop. You help with:
- Store policies and procedures
- Hours and locations
- General product information
- Account questions
- Navigation and website help

Be friendly, concise, and helpful. Route to specialists if needed.
    """)
    
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    
    print(f"💬 Response: {response.content[:100]}...")
    
    return {
        "messages": [AIMessage(content=response.content)],
        "response": response.content,
        "route_taken": "general"
    }


def handle_escalation(state: RoutingState) -> dict:
    """
    Handler for escalations to human agents
    """
    print(f"\n{'='*60}")
    print("🚨 ESCALATION HANDLER")
    print(f"{'='*60}")
    
    # For escalations, we create a ticket and notify
    escalation_message = f"""
I understand this is an important issue that needs special attention.
I'm escalating your case to our senior support team right now.

A specialist will contact you within the next 2 hours via:
- Email (to your registered email)
- Phone (if provided)

Your escalation ticket: ESC-{hash(state['user_input']) % 10000}

In the meantime, is there anything else I can help you with?
    """
    
    print(f"💬 Escalating to human agent...")
    
    return {
        "messages": [AIMessage(content=escalation_message)],
        "response": escalation_message,
        "route_taken": "escalation"
    }


# ============================================================================
# CONCEPT 4: Routing Logic (Conditional Edge Function)
# ============================================================================


def route_query(state: RoutingState) -> Literal["billing", "technical", "shipping", "general", "escalation"]:
    """
    Routing function that determines which handler to use
    
    This function is used in add_conditional_edges to decide the graph flow.
    
    Args:
        state: Current state with intent classification
        
    Returns:
        Name of the node to route to
    """
    intent = state.get("intent", "general")
    confidence = state.get("confidence", 0.0)
    
    print(f"\n🔀 ROUTING: Intent={intent}, Confidence={confidence:.2f}")
    
    # If confidence is low, route to general
    if confidence < 0.6:
        print("   ⚠️  Low confidence -> routing to GENERAL")
        return "general"
    
    # Map intents to handlers
    intent_map = {
        "billing": "billing",
        "technical": "technical",
        "shipping": "shipping",
        "general": "general",
        "escalation": "escalation"
    }
    
    route = intent_map.get(intent, "general")
    print(f"   ✅ Routing to: {route.upper()}")
    
    return route


# ============================================================================
# CONCEPT 5: Building the Routing Graph
# ============================================================================


def create_routing_graph():
    """
    Create a graph with conditional routing:
    
    START -> Classify Intent -> [Route based on intent]
                                   |
                                   ├-> Billing Handler -> END
                                   ├-> Technical Handler -> END
                                   ├-> Shipping Handler -> END
                                   ├-> General Handler -> END
                                   └-> Escalation Handler -> END
    """
    # Initialize graph
    graph = StateGraph[RoutingState, None, RoutingState, RoutingState](RoutingState)
    
    # Add classification node
    graph.add_node("classify", classify_intent)
    
    # Add specialized handler nodes
    graph.add_node("billing", handle_billing)
    graph.add_node("technical", handle_technical)
    graph.add_node("shipping", handle_shipping)
    graph.add_node("general", handle_general)
    graph.add_node("escalation", handle_escalation)
    
    # Start with classification
    graph.add_edge(START, "classify")
    
    # Conditional routing from classify to handlers
    graph.add_conditional_edges(
        "classify",  # Source node
        route_query,  # Routing function
        {
            "billing": "billing",
            "technical": "technical",
            "shipping": "shipping",
            "general": "general",
            "escalation": "escalation"
        }
    )
    
    # All handlers lead to END
    graph.add_edge("billing", END)
    graph.add_edge("technical", END)
    graph.add_edge("shipping", END)
    graph.add_edge("general", END)
    graph.add_edge("escalation", END)
    
    return graph.compile()


# ============================================================================
# CONCEPT 6: Running Routing Examples
# ============================================================================


def run_routing_examples():
    """
    Test the routing system with different query types
    """
    app = create_routing_graph()
    
    # Test queries for each intent
    test_queries = [
        ("I was charged twice for my order!", "billing"),
        ("My laptop won't turn on after the update", "technical"),
        ("Where is my package? I ordered 5 days ago", "shipping"),
        ("What are your store hours?", "general"),
        ("This is ridiculous! I want to speak to a manager NOW!", "escalation"),
    ]
    
    for i, (query, expected_intent) in enumerate(test_queries, 1):
        print(f"\n{'#'*60}")
        print(f"TEST CASE {i}")
        print(f"{'#'*60}")
        
        state = {
            "user_input": query,
            "intent": "",
            "confidence": 0.0,
            "messages": [],
            "response": "",
            "route_taken": ""
        }
        
        result = app.invoke(state)
        
        print(f"\n{'─'*60}")
        print(f"RESULT SUMMARY")
        print(f"{'─'*60}")
        print(f"Query: {query}")
        print(f"Expected Intent: {expected_intent}")
        print(f"Detected Intent: {result['intent']}")
        print(f"Route Taken: {result['route_taken']}")
        print(f"Match: {'✅' if expected_intent == result['route_taken'] else '❌'}")


# ============================================================================
# BONUS: Confidence-Based Routing
# ============================================================================


def route_with_confidence(state: RoutingState) -> Literal["high_confidence", "low_confidence"]:
    """
    Example of routing based on confidence score
    """
    confidence = state.get("confidence", 0.0)
    
    if confidence >= 0.8:
        return "high_confidence"
    else:
        return "low_confidence"


def demo_confidence_routing():
    """
    Demonstrate routing based on confidence scores
    """
    print(f"\n{'='*60}")
    print("🎯 BONUS: Confidence-Based Routing")
    print(f"{'='*60}")
    
    class ConfidenceState(TypedDict):
        confidence: float
        route: str
    
    def high_confidence_handler(state: ConfidenceState) -> dict:
        return {"route": "Handled with automation"}
    
    def low_confidence_handler(state: ConfidenceState) -> dict:
        return {"route": "Escalated to human review"}
    
    graph = StateGraph[ConfidenceState, None, ConfidenceState, ConfidenceState](ConfidenceState)
    graph.add_node("high_confidence", high_confidence_handler)
    graph.add_node("low_confidence", low_confidence_handler)
    
    graph.add_conditional_edges(
        START,
        route_with_confidence,
        {
            "high_confidence": "high_confidence",
            "low_confidence": "low_confidence"
        }
    )
    
    graph.add_edge("high_confidence", END)
    graph.add_edge("low_confidence", END)
    
    app = graph.compile()
    
    # Test different confidence levels
    for conf in [0.95, 0.75, 0.50]:
        result = app.invoke({"confidence": conf, "route": ""})
        print(f"Confidence {conf:.2f} -> {result['route']}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found")
        exit(1)
    
    # Run routing examples
    run_routing_examples()
    
    # Demonstrate confidence routing
    demo_confidence_routing()
    
    print("\n" + "="*60)
    print("💡 KEY TAKEAWAYS")
    print("="*60)
    print("1. Conditional edges enable dynamic routing")
    print("2. Routing functions return node names as strings")
    print("3. Intent classification guides conversation flow")
    print("4. Specialized handlers improve response quality")
    print("5. Confidence scores enable fallback routing")
    print("="*60)
    
    print("\n🎓 Next Steps:")
    print("   Run 005-multi-tools.py to learn about")
    print("   combining routing with multiple tools!\n")


# ============================================================================
# COMPREHENSIVE EXPLANATION: CONDITIONAL ROUTING WITH REAL-WORLD EXAMPLE
# ============================================================================
"""
CONDITIONAL ROUTING EXPLAINED - REAL-WORLD EXAMPLE
==================================================

Imagine you're calling a large company's customer support line. Instead of 
connecting you to a random agent, the system intelligently routes your call 
to the right specialist based on what you need. This is exactly what 
conditional routing does in LangGraph!

REAL-WORLD SCENARIO: Customer Support Call Center
--------------------------------------------------

Scenario: Sarah calls TechShop customer support about a billing issue.

STEP 1: CUSTOMER CALLS IN
   Sarah: "Hi, I was charged twice for my order #12345. I need a refund!"

STEP 2: INTENT CLASSIFICATION (classify_intent node)
   The system analyzes Sarah's message:
   - Uses LLM to understand the intent
   - Classifies: "billing" (payment/refund issue)
   - Confidence: 0.95 (very confident)
   - Reason: "Customer mentions charge and refund - clear billing issue"
   
   State after classification:
   {
       "user_input": "I was charged twice for my order #12345...",
       "intent": "billing",
       "confidence": 0.95,
       "messages": [HumanMessage("I was charged twice...")]
   }

STEP 3: ROUTING DECISION (route_query function)
   This is where route_query() comes in! It's the "traffic controller" 
   that decides which specialist should handle Sarah's case.
   
   route_query() checks:
   1. Intent = "billing" ✓
   2. Confidence = 0.95 (high, > 0.6) ✓
   3. Decision: Route to "billing" handler
   
   Returns: "billing" (the node name to route to)

STEP 4: CONDITIONAL EDGE EXECUTES
   The graph's conditional edge uses route_query()'s return value:
   - route_query() returned "billing"
   - Conditional edge maps "billing" → "billing" node
   - Graph routes to handle_billing() node

STEP 5: SPECIALIZED HANDLER PROCESSES (handle_billing node)
   The billing specialist (node) handles Sarah's case:
   - Has specialized knowledge about refunds, charges, payments
   - Uses billing-specific system prompts
   - Provides expert-level response about refund process
   
   Response: "I understand your concern about the double charge. Let me 
            process your refund immediately. The refund will appear in 
            your account within 5-7 business days..."

STEP 6: COMPLETION
   Sarah receives expert help from the right specialist, and the 
   conversation ends successfully.

VISUAL FLOW:
-----------

    START
      │
      ↓
[classify_intent] ← Analyzes: "billing" (confidence: 0.95)
      │
      ↓
[route_query] ← Decision function: Returns "billing"
      │
      ├─→ "billing" → [handle_billing] → END ✓ (Sarah's case)
      ├─→ "technical" → [handle_technical] → END
      ├─→ "shipping" → [handle_shipping] → END
      ├─→ "general" → [handle_general] → END
      └─→ "escalation" → [handle_escalation] → END


WHY IS route_query() NEEDED?
============================

route_query() is ESSENTIAL because:

1. DECISION MAKING LOGIC
   - It's the "brain" that decides WHERE to route based on state
   - Without it, the graph wouldn't know which handler to use
   - It implements business logic (e.g., low confidence → general handler)

2. CONDITIONAL EDGE REQUIREMENT
   - Conditional edges REQUIRE a routing function
   - The function examines state and returns a node name
   - LangGraph uses this return value to route the flow

3. FLEXIBILITY AND CONTROL
   - You can add complex logic (confidence checks, fallbacks, etc.)
   - Example: If confidence < 0.6, route to general (safer default)
   - Can implement business rules (e.g., premium customers → premium handler)

4. TYPE SAFETY
   - Returns Literal type: ensures only valid node names
   - Prevents typos: "bilng" would be caught by type checker
   - Maps to actual node names in the graph

5. SEPARATION OF CONCERNS
   - Classification node: "What is the intent?" (classify_intent)
   - Routing function: "Where should we go?" (route_query)
   - Handler nodes: "How do we handle it?" (handle_billing, etc.)

WHAT HAPPENS WITHOUT route_query()?
-----------------------------------

Without route_query(), you'd have to:
- Use regular edges (always go to same place)
- Can't make dynamic decisions
- Can't route based on state
- Can't implement fallback logic

Example of what you CAN'T do without route_query():
   ❌ "If confidence is low, go to general handler"
   ❌ "If intent is billing, go to billing specialist"
   ❌ "If customer is premium, go to premium queue"

WITH route_query(), you CAN:
   ✅ All of the above! Dynamic routing based on any state condition


DETAILED BREAKDOWN OF route_query() FUNCTION
---------------------------------------------

def route_query(state: RoutingState) -> Literal["billing", ...]:
    # 1. Extract intent and confidence from state
    intent = state.get("intent", "general")  # Default to "general"
    confidence = state.get("confidence", 0.0)  # Default to 0.0
    
    # 2. BUSINESS LOGIC: Low confidence → safer default
    if confidence < 0.6:
        return "general"  # Don't trust low-confidence classifications
    
    # 3. MAP intent to handler node name
    intent_map = {
        "billing": "billing",      # billing intent → billing handler
        "technical": "technical",   # technical intent → technical handler
        # ... etc
    }
    
    # 4. Return the node name (must match a node in the graph!)
    return intent_map.get(intent, "general")  # Default to "general"


HOW CONDITIONAL EDGES USE route_query()
---------------------------------------

graph.add_conditional_edges(
    "classify",        # FROM this node
    route_query,      # USE this function to decide
    {                 # MAP return values to node names
        "billing": "billing",      # If route_query returns "billing" → go to "billing" node
        "technical": "technical",  # If route_query returns "technical" → go to "technical" node
        # ... etc
    }
)

Flow:
1. classify_intent() completes → state has intent="billing", confidence=0.95
2. Conditional edge calls route_query(state)
3. route_query() examines state, returns "billing"
4. Conditional edge looks up "billing" in mapping → routes to "billing" node
5. handle_billing() executes


ANALOGY: CALL CENTER ROUTING
-----------------------------

Think of route_query() like a call center operator:

Traditional Call Center:
   Operator: "Press 1 for billing, 2 for technical..."
   Customer presses button → routed to department

AI-Powered Call Center (This Code):
   AI Classifier: Analyzes what customer said → "billing" intent
   route_query(): Like operator deciding → "Send to billing department"
   Conditional Edge: Like phone system → Routes call to billing queue
   Handler Node: Like billing specialist → Handles the call


KEY TAKEAWAYS
-------------

1. route_query() is the DECISION FUNCTION
   - It examines state and decides where to route
   - Returns a node name (string literal)
   - Must match a node name in your graph

2. Conditional Edges REQUIRE a Routing Function
   - Can't have conditional routing without a decision function
   - The function's return value determines the path
   - Maps return values to actual node names

3. Separation of Concerns
   - classify_intent(): "What is the problem?" (analysis)
   - route_query(): "Where should it go?" (routing decision)
   - handle_*(): "How do we solve it?" (execution)

4. Business Logic Lives in route_query()
   - Confidence thresholds
   - Fallback strategies
   - Priority routing
   - Custom business rules

5. Type Safety Matters
   - Literal return type ensures valid node names
   - Prevents runtime errors from typos
   - Makes code self-documenting


REAL-WORLD APPLICATIONS
-----------------------

This pattern is used in:
- Customer support systems (like this example)
- Help desk ticketing systems
- Content moderation (route to different reviewers)
- Document processing (route to different processors)
- Multi-agent systems (route to specialized agents)
- Workflow automation (route based on document type)
- E-commerce (route orders to different fulfillment centers)

The pattern scales from simple 2-way routing to complex multi-path 
decision trees with hundreds of routes!


# ============================================================================
# WHY NOT USE TOOLS INSTEAD OF route_query()?
# ============================================================================

QUESTION: Why can't we use tools (like in lesson 003) where the LLM decides 
which tool to call, instead of using route_query() for conditional routing?

This is an excellent architectural question! Let's explore both approaches 
and understand when to use each.

TWO DIFFERENT PATTERNS
======================

PATTERN 1: Tool-Based Routing (Lesson 003)
-------------------------------------------
In tool-based routing, the LLM decides which tool to call:
- LLM analyzes the query
- LLM decides: "I need to call search_faq tool"
- Tool executes and returns result
- LLM processes tool result and responds

Example:
    User: "What's your return policy?"
    LLM: "I'll search the FAQ for you" → calls search_faq tool
    Tool: Returns FAQ answer
    LLM: "Based on our FAQ, here's our return policy..."

PATTERN 2: Conditional Edge Routing (Lesson 004)
--------------------------------------------------
In conditional routing, route_query() decides which NODE to execute:
- classify_intent() analyzes the query
- route_query() decides: "Route to billing handler"
- Entire billing handler node executes (with its own LLM)
- Handler returns final response

Example:
    User: "I was charged twice!"
    classify_intent(): "billing" intent
    route_query(): Returns "billing"
    handle_billing() node executes → specialized billing LLM responds


KEY DIFFERENCES
===============

1. GRANULARITY
   ──────────────────────────────────────────────────────────────
   Tools: Fine-grained actions (search FAQ, check order, etc.)
   Conditional Routing: Coarse-grained workflows (entire handler nodes)
   
   Tools are like "functions" - small, focused operations
   Handlers are like "departments" - complete workflows with their own LLMs

2. LLM INVOLVEMENT
   ──────────────────────────────────────────────────────────────
   Tools: LLM decides tool → Tool executes → LLM processes result
   Conditional Routing: LLM classifies → route_query() routes → Handler LLM responds
   
   Tools: LLM is in the loop for EVERY decision
   Conditional Routing: LLM classifies once, then specialized handler takes over

3. CONTROL AND PREDICTABILITY
   ──────────────────────────────────────────────────────────────
   Tools: LLM might call wrong tool, call multiple tools, or skip tools
   Conditional Routing: Deterministic routing based on explicit logic
   
   route_query() gives you CONTROL over routing logic
   Tools give LLM AUTONOMY but less control

4. COMPLEXITY OF HANDLERS
   ──────────────────────────────────────────────────────────────
   Tools: Simple functions that return data
   Conditional Routing: Complex nodes with their own LLMs, prompts, and logic
   
   Each handler (billing, technical, etc.) is a FULL workflow
   Tools are just data retrieval/action functions

5. COST AND LATENCY
   ──────────────────────────────────────────────────────────────
   Tools: Multiple LLM calls (decide tool → process result)
   Conditional Routing: One classification call, then handler executes
   
   Conditional routing can be MORE efficient for complex workflows


WHY NOT USE TOOLS FOR ROUTING?
==============================

You COULD use tools for routing, but here's why route_query() is better:

SCENARIO 1: Using Tools for Routing (Possible but Problematic)
───────────────────────────────────────────────────────────────

@tool
def route_to_billing_handler(query: str) -> str:
    # This would be a tool that routes to billing
    # But what does it return? How does it route?
    return "billing response"

@tool  
def route_to_technical_handler(query: str) -> str:
    return "technical response"

Problems:
1. LLM might call MULTIPLE tools (billing AND technical?)
2. LLM might call NO tools (just respond directly)
3. No control over routing logic (can't enforce confidence thresholds)
4. Tools return strings, not state updates
5. Can't easily implement fallback logic (low confidence → general)
6. Each tool would need its own LLM call, increasing cost

SCENARIO 2: Using route_query() (Current Approach - Better)
─────────────────────────────────────────────────────────────

def route_query(state: RoutingState) -> Literal["billing", ...]:
    # Explicit control over routing logic
    if confidence < 0.6:
        return "general"  # Enforce fallback
    return intent_map.get(intent, "general")

Benefits:
1. Deterministic routing (always follows same logic)
2. Can enforce business rules (confidence thresholds, etc.)
3. Single classification call, then direct routing
4. Each handler is a complete workflow (not just a tool)
5. Better separation: classification → routing → execution
6. More predictable and debuggable


WHEN TO USE TOOLS VS CONDITIONAL ROUTING
========================================

USE TOOLS WHEN:
✅ You need the LLM to decide between multiple simple actions
✅ Actions are data retrieval or simple operations
✅ You want LLM autonomy and flexibility
✅ Tools can be called multiple times in sequence
✅ Example: "Search FAQ, then check order status, then create ticket"

USE CONDITIONAL ROUTING WHEN:
✅ You need to route to COMPLETE WORKFLOWS (not just tools)
✅ You want explicit control over routing logic
✅ You need to enforce business rules (confidence, priority, etc.)
✅ Each route is a specialized handler with its own LLM/prompts
✅ Example: "Route billing queries to billing specialist workflow"

HYBRID APPROACH (Best of Both Worlds)
======================================

You can COMBINE both patterns:

1. Use conditional routing to route to specialized handlers
2. Each handler uses tools for specific actions

Example Flow:
    START
      ↓
    [classify_intent] → "billing" intent
      ↓
    [route_query] → "billing" handler
      ↓
    [handle_billing] → Uses tools:
                        - check_order_status tool
                        - process_refund tool
                        - create_ticket tool
      ↓
    END

This gives you:
- Controlled routing (conditional edges)
- Flexible actions (tools within handlers)
- Best of both worlds!


REAL-WORLD ANALOGY
==================

Tools = Individual Actions
──────────────────────────
Like a Swiss Army knife - each tool does one thing:
- Screwdriver tool → tighten screw
- Knife tool → cut something
- Scissors tool → cut paper

You decide which tool to use for each task.

Conditional Routing = Department Routing
─────────────────────────────────────────
Like routing a customer to the right department:
- Billing department → Handles ALL billing issues (has its own tools)
- Technical department → Handles ALL technical issues (has its own tools)
- Shipping department → Handles ALL shipping issues (has its own tools)

Each department is a COMPLETE WORKFLOW, not just a single tool.


CODE COMPARISON
===============

TOOL-BASED APPROACH (What you're asking about):
────────────────────────────────────────────────

@tool
def handle_billing(query: str) -> str:
    # This would be a tool
    llm = create_billing_llm()
    return llm.invoke(query)

# LLM decides: "I'll call handle_billing tool"
# Problem: LLM might not call it, or call multiple tools

CONDITIONAL ROUTING APPROACH (Current):
───────────────────────────────────────

def route_query(state) -> Literal["billing", ...]:
    # Explicit routing logic
    return "billing" if intent == "billing" else "general"

def handle_billing(state) -> dict:
    # This is a NODE, not a tool
    llm = create_billing_llm()
    response = llm.invoke(state["messages"])
    return {"response": response.content}

# route_query() ALWAYS routes correctly
# handle_billing() ALWAYS executes when routed


SUMMARY
=======

route_query() is used instead of tools because:

1. ROUTING IS DIFFERENT FROM ACTIONS
   - Routing = "Where should this go?" (decision)
   - Tools = "What should I do?" (action)
   - You want CONTROL over routing, AUTONOMY for actions

2. HANDLERS ARE COMPLETE WORKFLOWS
   - Each handler (billing, technical) is a full workflow
   - Not just a simple tool function
   - Has its own LLM, prompts, and logic

3. PREDICTABILITY AND CONTROL
   - route_query() gives explicit control
   - Can enforce business rules
   - Deterministic routing logic

4. EFFICIENCY
   - One classification call
   - Direct routing to handler
   - Handler executes complete workflow

5. SEPARATION OF CONCERNS
   - Classification: "What is it?" (classify_intent)
   - Routing: "Where does it go?" (route_query)
   - Execution: "How do we handle it?" (handle_*)

You CAN use tools WITHIN handlers (hybrid approach), but routing 
itself benefits from explicit control via route_query().
"""
