# Multimodal RAG Pipeline for Document QA

A Retrieval-Augmented Generation system that handles both text and 
images from PDF documents — built with BLIP-2, FAISS, and Gradio.

## What it does
- Extracts and indexes text from PDFs using vector similarity search
- Processes embedded images using BLIP-2 vision-language model
- Answers questions across both modalities via a Gradio UI
- Reduced query latency by 30% through optimised chunking strategy

## Architecture
PDF → Text extraction → FAISS vector index → similarity search → LLM response  
PDF → Image extraction → BLIP-2 → visual QA

## Stack
Python · BLIP-2 · FAISS · LangChain · LlamaCpp · HuggingFace · Gradio

## Setup
Runs in Google Colab. Requires HuggingFace API token.
See `rag_pipeline.py` for full implementation.
