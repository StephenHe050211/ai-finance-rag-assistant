# AI Finance RAG Assistant

## One-line summary
A lightweight retrieval-augmented assistant that answers finance/business questions using local documents and returns supporting source snippets.

## Why this project matters
Many companies cannot send internal financial documents to public AI tools. This project shows how an offline or private AI workflow can retrieve relevant context before answering.

## Skills demonstrated
- Text chunking and document retrieval
- TF-IDF similarity search as a transparent baseline
- Citation-style answer generation
- AI governance thinking: answers are grounded in retrieved documents

## How to run

```bash
python src/rag_assistant.py "What costs are deducted before AE rebate is calculated?"
```

## Suggested resume bullet
Created a local RAG-style finance assistant that retrieves relevant internal document chunks before generating grounded answers, reducing hallucination risk for finance operations use cases.
