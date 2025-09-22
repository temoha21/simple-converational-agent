# Simple Conversational Agent

This repository contains several Python scripts demonstrating the use of modern LLMs (Large Language Models) for conversational AI, text analysis, and data analysis tasks. The code leverages the LangChain framework, Google Gemini models, and other open-source tools to build interactive agents and workflows.

## Features

- **Conversational Chatbot with History**: Maintains chat history per user session, allowing for context-aware conversations.
- **Text Analysis Workflow**: Classifies, extracts entities, and summarizes input text using a graph-based approach.
- **Data Analysis Agent**: Interactively analyzes a synthetic car sales dataset using natural language queries.

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies
- Google Gemini API key (set as `API_KEY` in a `.env` file)

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd simple-converational-agent
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with your Google Gemini API key:
   ```env
   API_KEY=your_google_gemini_api_key
   ```

## Usage

### 1. Conversational Chatbot
Run:
```bash
python simple-chat-with-history.py
```
- Enter a unique username to start a session.
- Type your questions; type `exit` or `quit` to stop, or `logout` to switch users.

### 2. Text Analysis Workflow
Run:
```bash
python langgraph_basics.py
```
- Analyzes a sample text: classifies, extracts entities, and summarizes.
- Modify `sample_text` in the script to analyze your own text.

### 3. Data Analysis Agent
Run:
```bash
python simple_data_analysis_agent.py
```
- Interactively ask questions about a synthetic car sales dataset.
- Type `exit` or `quit` to stop.

## Project Structure

- `simple-chat-with-history.py` — Conversational chatbot with user history
- `langgraph_basics.py` — Text analysis workflow using LangGraph
- `simple_data_analysis_agent.py` — Data analysis agent with pandas DataFrame
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## Design Principles

- Clean, object-oriented code following SOLID, DRY, and KISS principles
- Modular and extensible design
- Example usage and context provided in each script

## License

MIT License

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Generative AI](https://ai.google/discover/gemini/)
- [Pandas](https://pandas.pydata.org/)

---
For questions or contributions, please open an issue or pull request.

