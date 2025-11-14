# 000 - Project Setup Guide

Welcome to the **LangGraph Customer Support Agent** tutorial! This guide will help you set up your development environment.

## Prerequisites

- **Python 3.11+** installed on your system
- **uv** package manager (we'll install this if you don't have it)
- **Google Generative AI API Key** (free to get)

## Step 1: Install UV Package Manager

UV is a fast Python package installer and resolver written in Rust.

### Windows (PowerShell)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:
```bash
uv --version
```

## Step 2: Get Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

**Note**: Google Generative AI offers a generous free tier perfect for learning!

## Step 3: Clone/Download Project

Navigate to your project directory:
```bash
cd "C:\Learning\GenAI Track\week-4"
```

## Step 4: Create Virtual Environment and Install Dependencies

```bash
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# Install all dependencies
uv pip install -e .
```

## Step 5: Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Google API key
# On Windows, you can use:
notepad .env
```

Update the `.env` file:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

## Step 6: Verify Installation

Create a test file `test_setup.py`:

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Test Google API connection
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    response = llm.invoke("Say 'Setup successful!' if you can read this.")
    print("✅ Setup verified!")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"❌ Setup failed: {e}")
```

Run the test:
```bash
uv run python test_setup.py
```

## Project Structure

```
week-4/
├── 000-project-setup.md          # This file
├── 001-basic-state.py             # Intro to State
├── 002-simple-chatbot.py          # Basic chatbot
├── 003-tools-integration.py       # Adding tools
├── 004-conditional-routing.py     # Conditional edges
├── 005-multi-tools.py             # Multiple tools
├── 006-memory-checkpointing.py    # Memory & persistence
├── 007-human-in-loop.py           # Human escalation
├── 008-advanced-patterns.py       # Subgraphs
├── src/                           # Production-ready system
│   ├── __init__.py
│   ├── agents/                    # Agent definitions
│   ├── tools/                     # Tool implementations
│   ├── state/                     # State management
│   ├── graph/                     # Graph construction
│   └── main.py                    # Entry point
├── pyproject.toml                 # Project dependencies
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
└── README.md                      # Main documentation
```

## Learning Path

Follow the numbered files in order:

1. **001** - Understanding State in LangGraph
2. **002** - Building a simple chatbot
3. **003** - Integrating tools (FAQ search)
4. **004** - Conditional routing based on intent
5. **005** - Working with multiple tools
6. **006** - Adding memory and checkpointing
7. **007** - Implementing human-in-the-loop
8. **008** - Advanced patterns with subgraphs
9. **src/** - Production-ready complete system

## Troubleshooting

### Issue: UV command not found
**Solution**: Restart your terminal after installing UV, or add it to PATH manually.

### Issue: Google API Key invalid
**Solution**: 
- Make sure you copied the entire key without spaces
- Verify the key is active in Google AI Studio
- Check that .env file is in the project root

### Issue: Module not found
**Solution**: 
```bash
# Make sure virtual environment is activated
.venv\Scripts\activate  # Windows
# Then reinstall
uv pip install -e .
```

### Issue: Permission errors on Windows
**Solution**: Run PowerShell as Administrator when installing UV.

## Next Steps

Once setup is complete, proceed to **001-basic-state.py** to begin learning LangGraph!

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Google Generative AI Docs](https://ai.google.dev/docs)
- [UV Documentation](https://docs.astral.sh/uv/)

---

**Ready to start? Let's build something amazing! 🚀**


