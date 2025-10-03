# app.py
"""
Clinical Video Retrieval Tool

- Configuration via environment variables (see README).
- Optional query variations via query_variations.generate_query_variations()
- Uses Flask's instance folder for lightweight local data (e.g., survey file).
"""

from flask import Flask, render_template, request, redirect, send_from_directory, url_for, abort
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from datetime import timedelta
import requests
import os
import math
import json
import torch
import time
from pathlib import Path

# Import the new query variation function (replaces old openai_gpt_functions import)
from query_variations import generate_query_variations

# Optional: moviepy for local clip previews (can be removed if you only serve pre-cut segments)
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Load .env if present
load_dotenv(override=False)

# Required / recommended env vars (see README)
SOLR_CORE_URL = os.getenv("SOLR_CORE_URL", "http://localhost:8983/solr/June2025BGE")
VIDEO_DIRECTORY = os.getenv("VIDEO_DIRECTORY", "")  # Directory holding original/full videos (optional)
VIDEO_URL_MAP_PATH = os.getenv("VIDEO_URL_MAP_PATH", "video_url_map.json")  # Maps file names to public URLs (optional)
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

# App & instance folder (Flask uses instance/ for local, non-checked-in data)
app = Flask(__name__, instance_relative_config=True)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure instance folder exists (for survey file, temp clips, etc.)
Path(app.instance_path).mkdir(parents=True, exist_ok=True)

# Lightweight local storage paths (not committed)
SURVEY_PATH = Path(app.instance_path) / "survey_responses.txt"
PREVIEW_DIR = Path(app.instance_path) / "previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------

def safe_read_json(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def convert_time_to_seconds(timestr):
    """Convert 'HH:MM:SS' or 'MM:SS' (or list with one of those) to seconds."""
    if isinstance(timestr, list):
        timestr = timestr[0] if timestr else "00:00:00"
    parts = str(timestr).split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h = 0
        m, s = map(int, parts)
    else:
        raise ValueError("Invalid time format")
    return int(timedelta(hours=h, minutes=m, seconds=s).total_seconds())

def find_segment(video_name, timestamp_str, segment_minutes=5):
    """Find a segmented file name like '<video_name>_<n>.mp4' that contains timestamp."""
    video_name_str = str(video_name).strip("[]")
    timestamp = convert_time_to_seconds(timestamp_str)
    segment_number = max(1, math.ceil(timestamp / (segment_minutes * 60)))
    return f"{video_name_str}_{segment_number}.mp4"

def construct_video_url(file_name_number):
    """
    Map an integer-ish video base name (e.g., '123') to a public/original URL using video_url_map.json.
    Falls back to 'URL not found' if missing.
    """
    file_name = f"{file_name_number}.mp4"
    url_map = safe_read_json(VIDEO_URL_MAP_PATH)
    return url_map.get(file_name, "URL not found")

def device_and_precision():
    """
    Choose device and safe precision. MPS doesn't support full FP16 everywhere.
    We'll avoid manual .half() unless it's truly supported to keep things stable.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps"), False  # fp16 False (safer on MPS)
    if torch.cuda.is_available():
        return torch.device("cuda"), True  # fp16 True on CUDA
    return torch.device("cpu"), False

# ------------------------------------------------------------------------------
# Model Loading (load once)
# ------------------------------------------------------------------------------

_models = {
    "bi_encoder": None,
    "tokenizer": None,
    "cross_encoder": None,
    "device": None,
    "use_fp16": False,
}

def load_models():
    if _models["bi_encoder"] is not None:
        return _models

    device, use_fp16 = device_and_precision()

    # Bi-encoder for embeddings
    bi_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=str(device))

    # Cross-encoder for reranking
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    cross_encoder = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
    cross_encoder.eval().to(device)
    # Avoid forcing fp16 on non-CUDA devices
    if use_fp16:
        cross_encoder.half()

    _models.update(
        bi_encoder=bi_encoder,
        tokenizer=tokenizer,
        cross_encoder=cross_encoder,
        device=device,
        use_fp16=use_fp16,
    )
    return _models

# ------------------------------------------------------------------------------
# Search functions
# ------------------------------------------------------------------------------

def keyword_search(q, desired_number_of_results=60, minimum_word_count=3):
    """
    Keyword search via Solr 'sentence' field using AND/OR fallback.
    """
    keywords = q.split()
    and_query = ' AND '.join(keywords)

    def get_filtered_results(query_str):
        params = {
            'q': f'sentence:({query_str})',
            'fl': 'id,timestamp,sentence,file_name',
            'wt': 'json',
            'rows': desired_number_of_results
        }
        try:
            response = requests.get(f"{SOLR_CORE_URL}/select", params=params, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print("Keyword search failed:", e)
            return []
        docs = response.json().get('response', {}).get('docs', [])
        filtered = []
        for d in docs:
            sentence = d.get('sentence', "")
            if isinstance(sentence, list):
                sentence = ' '.join(sentence)
            if len(sentence.split()) >= minimum_word_count:
                filtered.append(d)
        return filtered

    res = get_filtered_results(and_query)
    if len(res) < desired_number_of_results and keywords:
        or_query = ' OR '.join(keywords)
        or_res = get_filtered_results(or_query)
        # dedup by id
        seen = {d['id'] for d in res}
        res.extend([d for d in or_res if d['id'] not in seen])
    return res

def knn_search(q, desired_number_of_results=60):
    models = load_models()
    bi_encoder = models["bi_encoder"]

    try:
        q_vector = bi_encoder.encode(q)
    except Exception as e:
        print("Embedding encode failed:", e)
        return []

    q_vector_str = f"[{','.join(map(str, list(map(float, q_vector))))}]"
    params = {
        "fl": "id,timestamp,sentence,file_name,score",
        "rows": str(desired_number_of_results),
        "wt": "json",
        "q": f"{{!knn f=vector topK={desired_number_of_results}}}{q_vector_str}"
    }
    # KNN uses POST with JSON payload on newer Solr setups; we keep GET fallback here for compatibility.
    try:
        response = requests.get(f"{SOLR_CORE_URL}/select", params=params, timeout=20)
        response.raise_for_status()
        return response.json().get('response', {}).get('docs', [])
    except Exception as e:
        print("kNN search failed:", e)
        return []

# ------------------------------------------------------------------------------
# Combined search + rerank + interleave
# ------------------------------------------------------------------------------

def combined_search_interleaved(
    query,
    desired_number_of_results=60,
    minimum_word_count=3,
    final_limit=30,
    use_gpt_variations=True,
):
    timing = {}
    models = load_models()
    tokenizer = models["tokenizer"]
    cross_encoder = models["cross_encoder"]
    device = models["device"]

    # Step 1: Query variations (optional)
    start_gpt = time.perf_counter()
    queries_to_run = [query]
    if use_gpt_variations:
        # Uses the new function from query_variations.py (env-driven; safe fallback)
        queries_to_run = list({qv for qv in generate_query_variations(query, n_variations=10) if qv.strip()})
    timing['gpt_query_variations'] = time.perf_counter() - start_gpt

    # Step 2: Search & merge for each query
    start_search = time.perf_counter()
    query_results_list = []
    for q in queries_to_run:
        keyw_docs = keyword_search(q, desired_number_of_results, minimum_word_count)
        knn_docs = knn_search(q, desired_number_of_results)
        # Deduplicate by id
        unique_docs = list({d['id']: d for d in keyw_docs + knn_docs}.values())
        query_results_list.append(unique_docs)
    timing['search_and_merge'] = time.perf_counter() - start_search

    # If nothing found, return early
    if not any(query_results_list):
        return []

    # Step 3: Batch Cross-Encoder Reranking
    start_rerank = time.perf_counter()
    all_pairs = []
    doc_metadata = []
    for q_idx, q in enumerate(queries_to_run):
        for doc in query_results_list[q_idx]:
            doc_text = ' '.join(doc['sentence']) if isinstance(doc.get('sentence'), list) else doc.get('sentence', '')
            all_pairs.append((q, doc_text))
            doc_metadata.append((q_idx, doc))

    # Tokenize
    start_tokenize = time.perf_counter()
    enc = tokenizer(all_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    timing['tokenization'] = time.perf_counter() - start_tokenize

    # Inference
    batch_size = 32
    scores = []
    start_inference = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(all_pairs), batch_size):
            batch = {k: v[start:start + batch_size] for k, v in enc.items()}
            logits = cross_encoder(**batch).logits.view(-1).float()
            scores.extend(logits.detach().cpu().tolist())
    timing['inference'] = time.perf_counter() - start_inference

    # Assign scores back
    for (q_idx, doc), score in zip(doc_metadata, scores):
        doc['cross_encoder_score'] = float(score)

    # Sort within each query
    for docs in query_results_list:
        docs.sort(key=lambda d: d.get('cross_encoder_score', 0.0), reverse=True)
    timing['batch_cross_encoder_rerank'] = time.perf_counter() - start_rerank

    # Step 4: Interleave
    start_interleave = time.perf_counter()
    final_results, seen = [], set()
    max_len = max((len(docs) for docs in query_results_list), default=0)
    for i in range(max_len):
        block = sorted(
            [docs[i] for docs in query_results_list if i < len(docs)],
            key=lambda d: d.get('cross_encoder_score', 0.0),
            reverse=True
        )
        for doc in block:
            _id = doc.get('id')
            if _id and _id not in seen:
                final_results.append(doc)
                seen.add(_id)
                if len(final_results) >= final_limit:
                    break
        if len(final_results) >= final_limit:
            break
    timing['interleaving'] = time.perf_counter() - start_interleave

    # Step 5: Enhance docs with URLs
    start_enhance = time.perf_counter()
    for doc in final_results:
        file_name_number = doc.get('file_name')
        if isinstance(file_name_number, list):
            file_name_number = file_name_number[0]
        ts = doc.get('timestamp')
        if isinstance(ts, list):
            ts = ts[0]

        if file_name_number and ts:
            try:
                segment_file = find_segment(file_name_number, ts)
                # This assumes you have static segments under /static/video_cuts/<segment.mp4>
                doc['video_segment_url'] = url_for('static', filename=f"video_cuts/{segment_file}")
                doc['original_video_url'] = construct_video_url(file_name_number)
            except Exception:
                doc['video_segment_url'] = None
                doc['original_video_url'] = construct_video_url(file_name_number)
        else:
            doc['video_segment_url'] = None
            doc['original_video_url'] = "URL not found"
    timing['enhance_docs'] = time.perf_counter() - start_enhance

    # Print timing (safe for dev; silence in production if needed)
    total_time = sum(timing.values())
    print("\nTiming Breakdown:")
    for step, t in timing.items():
        pct = (t / total_time * 100) if total_time > 0 else 0.0
        print(f"{step:<30}: {t:.2f}s ({pct:.1f}%)")
    print(f"{'TOTAL TIME':<30}: {total_time:.2f}s\n")

    return final_results

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = (request.form.get('query') or "").strip()
        if not query:
            return render_template('index.html', error="Please enter a query.", convert_time_to_seconds=convert_time_to_seconds)
        results = combined_search_interleaved(query)
        return render_template('results.html', results=results, convert_time_to_seconds=convert_time_to_seconds)
    return render_template('index.html', convert_time_to_seconds=convert_time_to_seconds)

@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    usefulness = request.form.get('usefulness', '')
    experience = request.form.get('experience', '')
    improvements = request.form.get('improvements', '')

    try:
        with open(SURVEY_PATH, 'a', encoding='utf-8') as f:
            f.write(f'Usefulness: {usefulness}\n')
            f.write(f'Experience: {experience}\n')
            f.write(f'Improvements: {improvements}\n')
            f.write('---\n')
    except Exception as e:
        print("Failed to write survey:", e)
    return redirect(url_for('index'))

@app.route('/preview/<file_name>/<int:timestamp>')
def preview(file_name, timestamp):
    """
    Optional route: generate a short preview clip starting at `timestamp` (seconds).
    Requires MOVIEPY_AVAILABLE and VIDEO_DIRECTORY to be set.
    """
    if not MOVIEPY_AVAILABLE or not VIDEO_DIRECTORY:
        abort(404, description="Preview generation is not enabled.")

    # Sanitize inputs
    base = Path(file_name.strip("[]")).with_suffix(".mp4").name
    full_path = Path(VIDEO_DIRECTORY) / base
    if not full_path.exists():
        abort(404, description="Video not found.")

    # Extract a 3-minute clip into instance/previews
    try:
        duration = 180
        with VideoFileClip(str(full_path)) as video:
            end_time = min(timestamp + duration, int(video.duration))
            clip = video.subclip(timestamp, end_time)
            out_name = f"preview_{base}"
            out_path = PREVIEW_DIR / out_name
            clip.write_videofile(str(out_path))
    except Exception as e:
        print("Preview extraction failed:", e)
        abort(500, description="Failed to generate preview.")

    return send_from_directory(PREVIEW_DIR, f"preview_{base}", as_attachment=False)

@app.route('/video/<filename>')
def video(filename):
    # Render a simple player that can stream from static/video_cuts or an external URL
    return render_template('video_player.html', filename=filename)

# ------------------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Do NOT hardcode debug True for public repos.
    app.run(host="127.0.0.1", port=5000, debug=FLASK_DEBUG)
