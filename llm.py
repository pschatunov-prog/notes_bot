import os
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import torch

load_dotenv()

# Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "microsoft/phi-2")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Initialize models (lazy loading)
_llm_pipeline = None
_whisper = None
_embedder = None

def get_llm():
    """Lazy load LLM model."""
    global _llm_pipeline
    if _llm_pipeline is None:
        logging.info(f"Loading LLM model: {LLM_MODEL_NAME}")
        _llm_pipeline = pipeline(
            "text-generation",
            model=LLM_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="auto",
            max_new_tokens=256
        )
    return _llm_pipeline

def get_whisper():
    """Lazy load Whisper model."""
    global _whisper
    if _whisper is None:
        logging.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        _whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper

def get_embedder():
    """Lazy load sentence transformer for embeddings."""
    global _embedder
    if _embedder is None:
        logging.info("Loading sentence transformer")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

async def summarize_and_tag(text: str) -> dict:
    """
    Summarizes the text and generates tags using local LLM.
    Returns a dictionary with 'summary' and 'tags'.
    """
    prompt = f"""Analyze this note and provide a brief summary (max 1 sentence) and 3 relevant tags.
Return JSON format: {{"summary": "...", "tags": ["tag1", "tag2", "tag3"]}}

Note: {text}

JSON:"""
    
    try:
        llm = get_llm()
        response = llm(
            prompt,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
            return_full_text=False
        )
        
        content = response[0]['generated_text'].strip()
        
        # Try to extract JSON
        if '{' in content and '}' in content:
            json_start = content.index('{')
            json_end = content.rindex('}') + 1
            json_str = content[json_start:json_end]
            result = json.loads(json_str)
            return result
        else:
            # Fallback
            return {"summary": text[:100], "tags": []}
    except Exception as e:
        logging.error(f"Error in summarize_and_tag: {e}")
        return {"summary": text[:100], "tags": []}

async def semantic_search(query: str, notes_list: list) -> str:
    """
    Uses sentence embeddings to find relevant notes based on a query.
    """
    if not notes_list:
        return "No notes found."

    try:
        embedder = get_embedder()
        
        # Create embeddings for query and notes
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        note_texts = [f"{n['content']} {n['tags']}" for n in notes_list]
        note_embeddings = embedder.encode(note_texts, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = util.cos_sim(query_embedding, note_embeddings)[0]
        
        # Get top 3 results
        top_results = similarities.topk(min(3, len(notes_list)))
        
        if top_results.values[0] < 0.3:
            return "No relevant notes found for your query."
        
        result_text = "Found relevant notes:\n\n"
        for idx, score in zip(top_results.indices, top_results.values):
            note = notes_list[idx]
            result_text += f"📌 ID: {note['id']} (Score: {score:.2f})\n"
            result_text += f"Content: {note['content'][:100]}...\n"
            result_text += f"Tags: {note['tags']}\n\n"
        
        return result_text
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return f"Error during search: {e}"

async def analyze_notes(notes_text: str) -> str:
    """
    Analyzes all notes to generate ideas or plans using local LLM.
    """
    if not notes_text:
        return "No notes to analyze."

    prompt = f"""Based on these notes, provide a summary of main topics and 3 actionable suggestions:

{notes_text[:1500]}

Analysis:"""

    try:
        llm = get_llm()
        response = llm(
            prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )
        result = response[0]['generated_text'].strip()
        # Truncate if still too long (Telegram limit is 4096 chars)
        if len(result) > 4000:
            result = result[:4000] + "..."
        return result
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return f"Error during analysis: {e}"

async def transcribe_audio(file_path: str) -> str:
    """
    Transcribes audio using faster-whisper (local Whisper implementation).
    """
    try:
        whisper = get_whisper()
        segments, info = whisper.transcribe(file_path, beam_size=5)
        
        # Combine all segments
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return ""
