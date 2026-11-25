
import os
import uuid
import gradio as gr
import chromadb

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Please configure GEMINI_API_KEY in your .env file.")

client = genai.Client(api_key=API_KEY)

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="document_knowledge"
)


def chunk_text(text, size=1000, overlap=150):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def get_embeddings(texts, task_type):
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )
    return [item.values for item in response.embeddings]


def upload_documents(files):
    if not files:
        return "Please select at least one PDF."

    total_chunks = 0

    for file_path in files:
        reader = PdfReader(file_path)
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        chunks = chunk_text(text)

        if not chunks:
            continue

        embeddings = get_embeddings(
            chunks, "RETRIEVAL_DOCUMENT"
        )

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"source": os.path.basename(file_path)}
            for _ in chunks
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_chunks += len(chunks)

    if total_chunks == 0:
        return "No extractable text found in the uploaded PDFs."

    return f"Successfully processed {total_chunks} text chunks!"


def ask_question(question, history):
    if not question.strip():
        return "", history

    if collection.count() == 0:
        answer = "Please upload a PDF document first."
    else:
        query_embedding = get_embeddings(
            [question], "RETRIEVAL_QUERY"
        )[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(4, collection.count())
        )

        documents = results["documents"][0]
        sources = results["metadatas"][0]

        context = "\n\n".join(
            f"Source: {source['source']}\n{document}"
            for document, source in zip(documents, sources)
        )

        prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the supplied context.
If the answer is not available in the context, clearly say:
"I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}
"""

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt
        )

        answer = response.text or "No answer was generated."

    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    return "", history


with gr.Blocks(title="Chat With Your Documents") as demo:
    gr.Markdown(
        "# 📚 Chat With Your Documents\n"
        "Upload PDFs and ask questions using RAG and Gemini."
    )

    files = gr.File(
        label="Upload PDF Documents",
        file_count="multiple",
        file_types=[".pdf"],
        type="filepath"
    )

    upload_button = gr.Button(
        "Process Documents",
        variant="primary"
    )

    upload_status = gr.Textbox(
        label="Processing Status",
        interactive=False
    )

    chatbot = gr.Chatbot(
        label="Document Chat",
        height=400
    )

    question = gr.Textbox(
        label="Ask a question",
        placeholder="Ask something about your documents..."
    )

    send_button = gr.Button("Ask Gemini")

    upload_button.click(
        upload_documents,
        inputs=files,
        outputs=upload_status
    )

    send_button.click(
        ask_question,
        inputs=[question, chatbot],
        outputs=[question, chatbot]
    )

    question.submit(
        ask_question,
        inputs=[question, chatbot],
        outputs=[question, chatbot]
    )


if __name__ == "__main__":
    demo.launch()
