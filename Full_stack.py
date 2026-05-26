# Reset file permissions (run before everything else)
!chmod -R 777 /content
!mkdir -p /content/images
!mkdir -p /content/vector_db
!chmod -R 777 /content/images
!chmod -R 777 /content/vector_db

!pip install -q -r Latese_requirements.txt
!pip install pypdf==3.17.4

# 1) system packages first  (you already have these lines)
!apt-get -qq update
!apt-get -qq install -y tesseract-ocr ghostscript

!pip install -q pymupdf pytesseract torch transformers langchain langchain_community chromadb gradio pillow sentence-transformers

!pip install -q langchain-huggingface

# Run these commands if DB issues persist
!rm -rf /content/vector_db
!rm -rf /content/chroma  # Sometimes Chroma creates this directory too
!mkdir -p /content/vector_db
!chmod 777 /content/vector_db  # Ensure full permissions

import io
import os
import shutil
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF
import gradio as gr
import pytesseract
import torch
from PIL import Image
from tqdm.auto import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint  # Updated import
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ================================================================
# 1) CONFIGURATION
# ================================================================
EMBED_MODEL = "intfloat/e5-large-v2"
LLM_MODEL = "google/flan-t5-large"
VLM_MODEL = "Salesforce/blip2-opt-2.7b"

DB_DIR = Path("/content/vector_db")
IMG_DIR = Path("/content/images")
IMG_DIR.mkdir(exist_ok=True, parents=True)

# Add API token handling
HF_API_TOKEN = None  # Will be set via UI

# ================================================================
# 2) VQA FUNCTIONALITY - GPU/CPU Compatible
# ================================================================
processor_vqa = vlm = None

def load_vlm():
    """Load the BLIP-2 model with appropriate settings for the current environment"""
    global processor_vqa, vlm
    if processor_vqa is None:
        print("🔄 Loading BLIP-2 model (first run takes ~1 minute)...")
        processor_vqa = Blip2Processor.from_pretrained(VLM_MODEL)

        # Check if CUDA is available
        has_cuda = torch.cuda.is_available()
        print(f"CUDA available: {has_cuda}")

        # Load model based on available hardware
        if has_cuda:
            # GPU mode - use half precision
            vlm = Blip2ForConditionalGeneration.from_pretrained(
                VLM_MODEL,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            # CPU mode - use FP32 and explicitly set device to CPU
            vlm = Blip2ForConditionalGeneration.from_pretrained(
                VLM_MODEL,
                device_map={"": "cpu"}
            )

        device_name = "CPU" if not has_cuda else f"GPU ({torch.cuda.get_device_name(0)})"
        print(f"✅ BLIP-2 model loaded on {device_name}")

def ask_about_image(img_path: Path, question: str):
    """Process an image with detailed error handling and debugging"""
    try:
        load_vlm()
        print(f"📄 Processing image: {img_path}")
        print(f"❓ Question: {question}")

        # Load and process the image
        img = Image.open(img_path).convert("RGB")
        print(f"🖼️ Image dimensions: {img.size}")

        # Format question with prompt template for better results
        prompted_question = f"Question: {question} Answer:"

        # Generate VQA response
        inputs = processor_vqa(img, prompted_question, return_tensors="pt").to(vlm.device)
        print(f"⚙️ Running inference on {vlm.device}...")

        # More appropriate generation parameters
        output = vlm.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=1.0
        )

        answer = processor_vqa.decode(output[0], skip_special_tokens=True)

        # Clean up the answer (remove the question part if it's repeated)
        if prompted_question in answer:
            answer = answer.split(prompted_question)[-1].strip()

        print(f"💬 Answer: {answer}")
        return answer
    except Exception as e:
        print(f"❌ VQA Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return f"Error processing image: {type(e).__name__}: {e}"

# ================================================================
# 3) PDF → TEXT/IMAGES  (PyMuPDF-only, robust)
# ================================================================
def extract_images(pdf_path: Path):
    doc = fitz.open(pdf_path)
    for page_idx, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            out_path = IMG_DIR / f"{pdf_path.stem}_p{page_idx+1}_{img_idx}.{base['ext']}"
            with open(out_path, "wb") as f:
                f.write(base["image"])
            yield page_idx, out_path

def parse_pdf(pdf_path: Path):
    try:
        doc = fitz.open(pdf_path)
        n_pages = doc.page_count

        # Pre-collect images
        img_map = {i: [] for i in range(n_pages)}
        for p, img_path in extract_images(pdf_path):
            img_map[p].append(str(img_path))

        pages = []
        for i in range(n_pages):
            text = doc[i].get_text("text").strip()
            if not text:  # OCR fallback
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes()))
                text = pytesseract.image_to_string(img).strip()

            pages.append(
                {"text": text,
                "images": img_map[i],
                "page": i + 1,
                "source": pdf_path.name}
            )
        return pages
    except Exception as e:
        print(f"❌ PDF Parsing Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return []

def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs = []
    for p in pages:
        if p["text"]:
            for chunk in splitter.split_text(p["text"]):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"page": p["page"], "source": p["source"], "has_images": bool(p["images"])}
                ))
    return docs

# ================================================================
# 4) VECTOR STORE - IN-MEMORY APPROACH
# ================================================================
# Global variable to keep the in-memory database
VECTOR_DB = None

def ingest_pdfs(pdf_paths: List[Path], overwrite: bool = False):
    """Process PDFs using in-memory database to avoid permission issues"""
    global VECTOR_DB

    # Force cleanup of old DB files
    if DB_DIR.exists():
        try:
            shutil.rmtree(DB_DIR)
        except:
            pass

    chunks = []
    for path in tqdm(pdf_paths, desc="🔍 Parsing PDFs"):
        chunks.extend(chunk_documents(parse_pdf(path)))

    if not chunks:
        print("⚠️ No text content extracted from PDFs!")
        return False

    try:
        # Create in-memory embeddings to avoid file permission issues
        print(f"📥 Creating embeddings for {len(chunks)} text chunks...")
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        # Create purely in-memory database (no persist_directory)
        VECTOR_DB = Chroma.from_documents(chunks, embedding=embedder)

        # Optional: Save document contents separately as backup
        DB_DIR.mkdir(exist_ok=True, parents=True)
        with open(DB_DIR / "documents.txt", "w") as f:
            for chunk in chunks:
                f.write(f"--- PAGE {chunk.metadata['page']} OF {chunk.metadata['source']} ---\n")
                f.write(chunk.page_content)
                f.write("\n\n")

        print("✅ Created in-memory vector database")
        return True
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        traceback.print_exc()
        return False

def get_vectordb():
    """Return the in-memory vector database"""
    global VECTOR_DB
    if VECTOR_DB is None:
        print("⚠️ No vector database loaded in memory")
        return None
    return VECTOR_DB

def is_pdf_ingested():
    """Check if PDFs have been ingested by checking the in-memory database"""
    global VECTOR_DB
    return VECTOR_DB is not None

# ================================================================
# 5) QA & VQA CHAINS
# ================================================================
def build_qa_chain():
    """Create a QA chain with the LLM and vector database"""
    global HF_API_TOKEN, VECTOR_DB

    if VECTOR_DB is None:
        raise ValueError("No documents have been processed. Please upload a PDF first.")

    if not HF_API_TOKEN:
        raise ValueError("Hugging Face API token not provided. Please enter your token in the setup tab.")

    prompt = PromptTemplate(
        template=textwrap.dedent("""\
            You are a helpful assistant answering questions **strictly** from the supplied context.
            If the answer is not contained, say "I don't know.".

            Context:
            {context}

            Question: {question}
            Answer:"""),
        input_variables=["context", "question"])

    retriever = VECTOR_DB.as_retriever(search_kwargs={"k": 5})

    # Use OpenAI instead which is much more stable
    # We'll fall back to a local model if needed
    try:
        # Option 1: Try with a more direct HuggingFace approach
        from langchain_community.llms import HuggingFaceTextGenInference

        llm = HuggingFaceTextGenInference(
            inference_server_url="https://api-inference.huggingface.co/models/" + LLM_MODEL,
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        )
    except Exception as e:
        print(f"Error initializing HuggingFace: {e}")

        # Option 2: Fall back to a local T5 model which is small and works reliably
        from langchain.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

        print("Falling back to local T5-small model...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512
        )

        llm = HuggingFacePipeline(pipeline=pipe)

    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True)

# ================================================================
# 6) GRADIO UI
# ================================================================
uploaded_paths: List[Path] = []
qa_chain = None

def _to_path(f):
    if isinstance(f, str):              # File(type="filepath")
        return Path(f)
    if isinstance(f, dict):             # File(type="auto") in some cases
        return Path(f["path"])
    if hasattr(f, "name"):              # old binary object
        return Path(f.name)
    raise ValueError(f"Unknown file object: {f!r}")

def set_api_token(token):
    global HF_API_TOKEN
    if not token.strip():
        return "❗ Please enter a valid Hugging Face API token."

    HF_API_TOKEN = token.strip()
    return "✅ API token saved! You can now process PDFs and ask questions."

def handle_upload(files):
    global uploaded_paths, qa_chain
    try:
        if not files:
            return "❗ Please select a PDF."

        if not HF_API_TOKEN:
            return "❗ Please enter your Hugging Face API token in the Setup tab first."

        uploaded_paths = [_to_path(f) for f in files]
        success = ingest_pdfs(uploaded_paths, overwrite=True)

        if not success:
            return "⚠️ Could not extract any content from the PDFs. They might be empty or protected."

        qa_chain = build_qa_chain()

        # Count extracted images
        image_count = len(list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png")) + list(IMG_DIR.glob("*.jpeg")))

        return f"✅ PDFs processed & indexed! Extracted {image_count} images."
    except Exception as e:
        traceback.print_exc()
        return f"❌ {type(e).__name__}: {e}"

def answer_text(question):
    global qa_chain

    if not HF_API_TOKEN:
        return "❗ Please enter your Hugging Face API token in the Setup tab first."

    if not is_pdf_ingested():
        return "❗ Upload & process PDFs first."

    if qa_chain is None:
        try:
            qa_chain = build_qa_chain()
            if qa_chain is None:
                return "❗ Problem accessing the vector database. Try re-uploading PDFs."
        except Exception as e:
            return f"❌ Error building QA chain: {str(e)}"

    if not question.strip():
        return "Enter a question."

    try:
        # Updated to use invoke instead of __call__
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        sources = ", ".join(
            {f'{d.metadata["source"]} p.{d.metadata["page"]}' for d in result["source_documents"]}
        )
        return f"**Answer:** {answer}\n\n**Sources:** {sources}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ {type(e).__name__}: {e}"

def answer_img(img, question):
    if img is None or not question.strip():
        return "Upload an image and enter a question."

    # Save the uploaded image to a temp file
    tmp = Path(tempfile.mktemp(suffix=".png"))
    img.save(tmp)

    try:
        resp = ask_about_image(tmp, question)
        return f"**VQA:** {resp}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ {type(e).__name__}: {e}"
    finally:
        tmp.unlink(missing_ok=True)

def launch_ui():
    with gr.Blocks(title="📑 Multimodal RAG for PDFs") as demo:
        gr.Markdown("# 📑 Multimodal Retrieval-Augmented QA for PDFs")

        with gr.Tab("0️⃣ Setup"):
            gr.Markdown("### Enter your Hugging Face API token")
            gr.Markdown("You need a Hugging Face API token to use the text question-answering feature. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)")
            token_input = gr.Textbox(
                label="Hugging Face API Token",
                placeholder="hf_...",
                type="password"
            )
            token_button = gr.Button("Save Token")
            token_output = gr.Markdown()
            token_button.click(set_api_token, inputs=token_input, outputs=token_output)

        with gr.Tab("1️⃣ Upload & Process PDFs"):
            files = gr.File(file_types=[".pdf"],
                            file_count="multiple",
                            type="filepath",
                            label="Select PDF files")
            btn   = gr.Button("Process PDFs")
            out   = gr.Markdown()
            btn.click(handle_upload, inputs=files, outputs=out)

        with gr.Tab("2️⃣ Ask Text Questions"):
            q        = gr.Textbox(label="Question about the PDFs")
            btn_txt  = gr.Button("Answer")
            txt_out  = gr.Markdown()
            btn_txt.click(answer_text, inputs=q, outputs=txt_out)

        with gr.Tab("3️⃣ Ask Image Questions"):
            gr.Markdown("Upload an image from /content/images or any relevant screenshot.")
            img     = gr.Image(type="pil")
            img_q   = gr.Textbox(label="Question about this image")
            btn_vqa = gr.Button("Answer")
            vqa_out = gr.Markdown()
            btn_vqa.click(answer_img, inputs=[img, img_q], outputs=vqa_out)

                # Add warning about CPU-only performance
        if not torch.cuda.is_available():
            gr.Markdown("⚠️ **Note:** Running in CPU-only mode. Image questions will be very slow.")

    # Launch with debug=True to see errors in the notebook
    demo.launch(share=True, debug=True)

# Run the UI
if __name__ == "__main__":
    launch_ui()

