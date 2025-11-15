# 🤖 LangGraph Customer Support Agent - Zero to Hero Tutorial

[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-blue)](https://github.com/langchain-ai/langgraph)
[![Google AI](https://img.shields.io/badge/Google%20AI-Gemini-orange)](https://ai.google.dev/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A comprehensive, production-ready tutorial for learning **LangGraph** from basics to advanced concepts through building a real-world **Customer Support Agent** system.

## 📚 What You'll Learn

This tutorial takes you from **zero to hero** in LangGraph through 8 progressive lessons and a complete production system:

### Core Concepts Covered

✅ **State Management** - TypedDict, annotations, reducers  
✅ **LLM Integration** - Google Generative AI (Gemini)  
✅ **Tools** - Function calling, tool binding, execution  
✅ **Conditional Routing** - Intent classification, dynamic flows  
✅ **Multi-Agent Systems** - Specialized agents, coordination

## 🎯 Why This Tutorial?

### Real-World Use Case
Unlike toy examples, this tutorial builds a **production-ready customer support system** that handles:
- Billing inquiries and refunds
- Technical troubleshooting
- Order tracking and shipping
- FAQ and policy questions
- Intelligent routing and escalations

### Progressive Learning
Start with basics and build complexity:
1. **001** → Simple state management
2. **002** → Add LLM integration
3. **003** → Integrate tools
4. **004** → Add routing logic
5. **005** → Multi-tool orchestration

### Production-Ready Code
- Proper project structure
- Configuration management
- Error handling
- Type hints
- Comprehensive comments
- Best practices throughout

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Google AI API key ([Get one free](https://makersuite.google.com/app/apikey))
- 30 minutes for basic tutorials
- 2+ hours for complete system

### Installation

```bash
# 1. Install UV package manager (if not installed)
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone or navigate to this directory
cd path/to/this/directory

# 3. Create virtual environment and install dependencies
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

uv pip install -e .

# 4. Set up environment variables
# Create .env file with:
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### Verify Setup

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✅ Setup OK!' if os.getenv('GOOGLE_API_KEY') else '❌ API key missing')"
```

## 📖 Learning Path

### 🎓 Tutorial Series (001-008)

Follow in order for best learning experience:

| File | Topic | Key Concepts | Time |
|------|-------|--------------|------|
| **000-project-setup.md** | Setup Guide | Installation, configuration | 10 min |
| **001-basic-state.py** | State Management | TypedDict, reducers, annotations | 15 min |
| **002-simple-chatbot.py** | LLM Integration | Google AI, messages, streaming | 20 min |
| **003-tools-integration.py** | Tools | @tool decorator, binding, execution | 25 min |
| **004-conditional-routing.py** | Routing | Intent classification, conditional edges | 25 min |
| **005-multi-tools.py** | Multi-Tools | Tool orchestration, specialized routing | 30 min |

**Total Tutorial Time:** ~3.5 hours

### Running Tutorials

```bash
# Run tutorials in order
python 001-basic-state.py
python 002-simple-chatbot.py
python 003-tools-integration.py
# ... and so on
```

Each tutorial is:
- **Self-contained** - Run independently
- **Well-documented** - Extensive comments
- **Demonstrates concepts** - Clear examples
- **Builds on previous** - Progressive complexity

*Happy Learning! 🚀*


