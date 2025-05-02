# Internal Knowledge Base Chatbot (RAG)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://idkbchatbot.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![RAG](https://img.shields.io/badge/Architecture-RAG-orange)

An internal-facing AI chatbot that provides instant access to company documents (SOPs, manuals, policies) using Retrieval-Augmented Generation.

## Problem Statement
- Employees waste **19% of work hours** searching for information (Chui & Company, 2012)
- Over-reliance on senior staff for routine queries
- High turnover organizations need self-service solutions

## Solution
✅ **24/7 document access** via natural language  
✅ **Reduced hallucinations** with RAG architecture  
✅ **Streamlit UI** for non-technical users  

## Tech Stack
| Component       | Technology |
|-----------------|------------|
| LLM            | Google Gemini 1.5 |
| Vector Database| MongoDB Atlas |
| Backend        | Python (LangChain) |
| Frontend       | Streamlit |
| Evaluation     | ROUGE-L metrics |

## Installation
```bash
git clone https://github.com/yourusername/internal-rag-chatbot.git
cd internal-rag-chatbot
pip install -r requirements.txt
