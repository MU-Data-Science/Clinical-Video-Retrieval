# Clinical Video Retrieval Tool

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg) ![Framework](https://img.shields.io/badge/framework-Flask-green.svg) ![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

This project is a Flask-based web application that provides a powerful semantic search interface for a collection of medical video transcripts. It uses a hybrid search approach, combining keyword search (BM25) with dense neural search (Sentence Transformers), and leverages Large Language Models (LLMs) to enhance query understanding.

## 📋 Table of Contents

- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Prerequisites](#-prerequisites)
- [Setup and Installation](#-setup-and-installation)
- [Running the Application](#-running-the-application)
- [Publications](#-publications)
- [Acknowledgements](#-acknowledgements)

## ✨ Features

- **Hybrid Search**: Combines keyword search (BM25 via Apache Solr) and dense semantic search to retrieve relevant passages from video transcripts.
- **Advanced Re-ranking**: Uses a `bge-reranker-v2-m3` cross-encoder to refine search results for higher accuracy.
- **Query Rewriting**: Leverages an LLM to rephrase user queries for more comprehensive retrieval.
- **Video Previews**: Allows users to instantly view the video segment corresponding to a search result.

## 🛠️ Technology Stack

- **Backend**: Flask  
- **Search**: Apache Solr 9.x (for keyword and dense vector search)  
- **AI / ML**:  
  - **Sentence Transformers**: `BAAI/bge-large-en-v1.5` for embeddings.  
  - **Cross-Encoder**: `BAAI/bge-reranker-v2-m3` for re-ranking.  
  - **LLM**: Configured for OpenAI API (or any compatible provider).  
- **Video Processing**: ffmpeg  

## ✅ Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.8+  
- [Apache Solr](https://solr.apache.org/downloads.html) (version 9.x)  
- [ffmpeg](https://ffmpeg.org/download.html)  

*Note for Apple Silicon Users: The setup is configured to use PyTorch with MPS for hardware acceleration.*

## 🚀 Setup and Installation

Follow these steps precisely to get the project running.

### 1. Clone the Repository

```sh
git clone https://github.com/MU-Data-Science/Clinical-Video-Retrieval
cd Clinical-Video-Retrieval
```

### 2. Create and Activate a Python Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```sh
# Create the virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

This project uses several Python libraries. A `requirements.txt` file is provided for easy installation.

```sh
pip install -r requirements.txt
```

*Note: If you are on an Apple Silicon Mac, PyTorch with MPS support should be installed automatically. If not, please refer to the official PyTorch installation guide.*

### 4. Set Up Environment Variables

Create a `.env` file in the project's root directory. This file will store your secret keys and configuration paths.  
⚠️ Make sure to add `.env` to your `.gitignore` file to avoid committing secrets.

```sh
# Copy the example file
cp .env.example .env
```

Now, open the `.env` file and add your specific values:

```ini
# .env file
LLM_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
VIDEO_DIRECTORY="/path/to/your/videos"
SOLR_CORE_URL="http://localhost:8983/solr/June2025BGE"
```

### 5. Configure Apache Solr

The search functionality relies on a properly configured Solr core.

#### A. Create a Solr Core

If you haven't already, create a new core. You can name it `June2025BGE` or choose another name (remember to update it in your `.env` file).

#### B. Configure the Schema

Use the Solr Schema API to add the necessary field type and fields for vector storage. Run these commands from your terminal.

**Field Type for Vectors:**

```sh
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field-type": {
    "name": "knn_vector",
    "class": "solr.DenseVectorField",
    "vectorDimension": "1024",
    "similarityFunction": "cosine"
  }
}' http://localhost:8983/solr/June2025BGE/schema
```

*(Note: `vectorDimension` is set to 1024 for the `BAAI/bge-large-en-v1.5` model.)*

**Fields:**

```sh
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": [
    {"name": "file_name", "type": "string", "stored": true},
    {"name": "sentence", "type": "text_general", "stored": true},
    {"name": "timestamp", "type": "string", "stored": true},
    {"name": "vector", "type": "knn_vector", "indexed": true, "stored": true}
  ]
}' http://localhost:8983/solr/June2025BGE/schema
```

### 6. Index Your Data

Run the indexing script `text2vector.py` to process your transcripts, generate embeddings, and create a JSON file for Solr.

```sh
# This script will generate a JSON file for indexing
python text2vector.py

# Post the generated data to your Solr core
# Replace the path with the actual path to your generated JSON file
bin/post -c June2025BGE /path/to/generated/output.json
```

### 7. Configure Video URLs

Ensure the `video_url_map.json` file is present and correctly maps your video filenames to their accessible URLs (e.g., local paths or cloud storage links).

---

## 🖥️ Running the Application

Once the setup is complete, you can run the Flask application:

```sh
python app.py
```

The application will be available at:  
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

Open this URL in your web browser to start searching.

---

## 📖 Publications
1. Zhandi Liu, Mirna Becevic, Amy Braddock, Mihai Popescu, Eduardo J. Simoes, and Praveen Rao - **An Empirical Evaluation of Deep Learning Techniques for Clinical Video Retrieval**. In Proc. of 8th IEEE International Conference on Multimedia Information Processing and Retrieval (MIPR 2025), 8 pages, San Jose, CA. [[PDF]](https://drive.google.com/file/d/1twxy8q__LeYMgOGfMS7bdjnMLEeHuds2/view)

---

## Acknowledgements
This work was supported by the National Institute of Diabetes And Digestive And Kidney Diseases of the National Institutes of Health under Award Number P30DK092950.
