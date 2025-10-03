import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

def get_device():
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def is_timestamp(line):
    """Check if a line is a timestamp in format HH:MM:SS or MM:SS."""
    return bool(re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', line.strip()))

def parse_transcript(file_path, file_name, model, chunk_size=3, overlap=0):
    """
    Parses a transcript file, chunks the content, and generates vector embeddings.

    This function reads a transcript, segments it by timestamps, and then applies a
    sliding window to create overlapping chunks of text. Each chunk is then
    encoded into a vector embedding using the provided Sentence Transformer model.

    Args:
        file_path (str): The path to the transcript file.
        file_name (str): The base name of the file for metadata.
        model (SentenceTransformer): The loaded Sentence Transformer model.
        chunk_size (int): The number of sentences to include in each chunk.
        overlap (int): The number of sentences to overlap between chunks.

    Returns:
        list: A list of dictionaries, where each dictionary represents a chunk
              containing its ID, metadata, text, and vector embedding.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Skip the first two lines which might be metadata
            lines = file.readlines()[2:]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    # 1. Group lines by timestamp
    all_sentences = []
    current_timestamp = None
    current_text = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if is_timestamp(stripped_line):
            if current_timestamp and current_text:
                all_sentences.append({
                    'timestamp': current_timestamp,
                    'sentence': " ".join(current_text).strip()
                })
            current_timestamp = stripped_line
            current_text = []
        else:
            current_text.append(stripped_line)
    
    # Add the last block
    if current_timestamp and current_text:
        all_sentences.append({
            'timestamp': current_timestamp,
            'sentence': " ".join(current_text).strip()
        })

    # 2. Apply sliding window chunking
    chunked_data = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("Step size (chunk_size - overlap) must be positive.")

    for i in range(0, len(all_sentences), step):
        chunk = all_sentences[i : i + chunk_size]
        if not chunk:
            continue

        combined_text = " ".join([c['sentence'] for c in chunk]).strip()
        start_timestamp = chunk[0]['timestamp']

        # 3. Embed the combined text
        vector = model.encode(combined_text, convert_to_tensor=False)

        # 4. Store as a single record
        chunk_record = {
            'id': str(uuid.uuid4()),
            'file_name': file_name,
            'timestamp': start_timestamp,
            'sentence': combined_text,
            'vector': vector.tolist()
        }
        chunked_data.append(chunk_record)

    return chunked_data

def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Process video transcript files into vector embeddings using Sentence Transformers."
    )
    parser.add_argument(
        "--input-dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing transcript files."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True, 
        help="Path to the directory where JSON output files will be saved."
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default='BAAI/bge-large-en-v1.5', 
        help="Name of the Sentence Transformer model to use from HuggingFace."
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=3, 
        help="Number of sentences per chunk."
    )
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=0, 
        help="Number of overlapping sentences between chunks."
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Automatically select the best device
    device = get_device()
    print(f"Using device: {device}")

    # Initialize the model
    print(f"Loading model: {args.model_name}...")
    model = SentenceTransformer(args.model_name, device=device)
    print("Model loaded successfully.")

    # Process each file in the directory
    input_path = Path(args.input_dir)
    files_to_process = [f for f in input_path.iterdir() if f.is_file()]

    if not files_to_process:
        print(f"No files found in {args.input_dir}. Exiting.")
        sys.exit(0)

    for i, file_path in enumerate(files_to_process):
        try:
            print(f"Processing file {i+1}/{len(files_to_process)}: {file_path.name}...")
            file_name_base = file_path.stem
            
            transcript_data = parse_transcript(
                file_path, 
                file_name_base, 
                model, 
                args.chunk_size, 
                args.overlap
            )
            
            if not transcript_data:
                print(f"  -> No data generated for {file_path.name}. Skipping.")
                continue

            output_path = Path(args.output_dir) / f'{file_name_base}.json'
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(transcript_data, json_file, indent=2)

        except Exception as e:
            print(f"  -> Failed to process {file_path.name}: {e}")

    print("\nAll transcripts have been processed and saved as separate JSON files.")

if __name__ == "__main__":
    main()
