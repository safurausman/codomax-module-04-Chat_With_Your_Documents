# Chat With Your Documents

A PDF document chat application built using **Retrieval-Augmented Generation (RAG)**, **Google Gemini API**, and **ChromaDB**.

## Features

- Upload PDF documents.
- Extract and process document text.
- Split text into chunks for retrieval.
- Store document embeddings in ChromaDB.
- Ask questions about uploaded documents.
- Generate answers using Google Gemini.
- Maintain a conversational document chat interface.

## Technology Stack

- Python
- Gradio
- Google Gemini API
- ChromaDB
- RAG (Retrieval-Augmented Generation)

## Prerequisites

- Python installed
- Google Gemini API key

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/safurausman/codomax-module-04-Chat_With_Your_Documents.git
   ```

2. Navigate to the project directory:

   ```bash
   cd codomax-module-04-Chat_With_Your_Documents
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure your Gemini API key in a `.env` file:

   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

6. Run the application:

   ```bash
   python app.py
   ```

## Learning Project

Developed as part of **Codomax Solutions – Module 04: Chat With Your Documents**.

## Author

Safura Usman
