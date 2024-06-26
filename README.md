# Langchain Ollama RAG App
The following code was developed for this article on [Mastering RAG](https://brightjourneyai.com/mastering-rag-local-intelligent-apps-with-langchain-ollama/). 

In the article I've walked through the nitty-gritty of leveraging Large Language Models (LLMs) for practical, business use cases. We started with understanding the limitations of LLMs and how fine-tuning and Retrieval Augmented Generation (RAG) can address these issues. Then, we dived into the nitty-gritty of building a RAG application using open-source tools.

It's clear that while LLMs are powerful, they aren't without their shortcomings, especially when it comes to accessing current or proprietary data. But fear not, because with a bit of ingenuity and the right tools, you can turn these challenges into opportunities. The combination of fine-tuning and RAG, supported by open-source models and frameworks like Langchain, ChromaDB, Ollama, and Streamlit, offers a robust solution to making LLMs work for you.

## How to Run
Make sure LLM is downloaded and running with Ollama
Example : to pull a model : ollama pull llama3:8b-instruct-q8_0
To run the model : ollama run llama3:8b-instruct-q8_0 
You can check model activities by running : ollama serve

1. Ensure poetry is installed for dependency management
2. CD to application directory
3. Run `poetry install` to install all dependencies
4. Run `streamlit run .\langchain-rag-bot\app.py`
