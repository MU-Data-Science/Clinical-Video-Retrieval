# Clinical Video Retrieval Tool

This project is a Flask-based web application that provides a powerful semantic search interface for a collection of medical video transcripts. It leverages a hybrid search approach, combining traditional keyword search with modern dense neural semantic search, and uses a Large Language Model (LLM) to rephrase user query to retrieve more relevant search results.

## Features

-   **Hybrid Search**: Combines keyword search (via Apache Solr) and dense vector search to retrieve relevant passages from video transcripts.
-   **Advanced Re-ranking**: Uses a `bge-reranker-v2-m3` cross-encoder to re-rank and refine search results for higher accuracy.
-   **Video Previews**: Allows users to view the original video segment corresponding to a search result.
-   **User-Friendly Interface**: A simple, clean web interface for searching and viewing results.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

-   **Python 3.8+**
-   **Apache Solr 9.x**: For indexing and keyword/dense neural search.
-   **ffmpeg**: A command-line tool for video processing, required for generating video previews.
-   **(For Apple Silicon Users)**: The application is configured to use PyTorch with MPS for hardware acceleration on Apple Silicon Macs.

## Setup and Installation

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

### 2. Create a Python Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

### 3. Install Dependencies

This project uses several Python libraries. A `requirements.txt` file is provided for easy installation.

*(Note: If you are on an Apple Silicon Mac, PyTorch with MPS support should be installed automatically. If not, please refer to the official PyTorch installation guide).*

### 4. Configure Apache Solr

The search functionality relies on a properly configured Solr core.

**A. Create a Solr Core:**
Create a new core named `June2025BGE` (or update the core name in `app.py`).

**B. Configure the Schema:**
You need to update the schema for the `June2025BGE` core to support dense vector search. Edit the `managed-schema` file (or use the Schema API) to add a field type for vectors and a field that uses it.

-   **Field Type for Vectors:**
    
    *(The `vectorDimension` of `1024` is required for the `BAAI/bge-large-en-v1.5` model used in this project.)*

-   **Fields:**
    Ensure your schema includes the following fields:
    - '*(field name="file_name" type="plongs" indexed="true" uninvertible="true" stored="true"/)*'
    - '*(field name="id" type="string" multiValued="false" indexed="true" required="true" stored="true"/)*'
    - '*(field name="sentence" type="text_general" indexed="true" uninvertible="true" stored="true"/)*'
    - '*(field name="timestamp" type="text_general" indexed="true" uninvertible="true" stored="true"/)*'
    - '*(field name="vector" type="knn_vector" indexed="true" uninvertible="true" stored="true"/)*'
    
**C. Index Your Data:**
Run the indexing script 'text2vector.py'. The script will:
1.  Read your video transcripts.
2.  Generate vector embeddings for each text segment using the `BAAI/bge-large-en-v1.5` model. Or change the model_name to the Sentence Transformer you prefer.
Post the data (ID, text, timestamp, filename, and vector) to your Solr core with Command:
-  *(bin/post -c June2025BGE /path/to/your/vector embeddings/BAAIbgeLargejson2025)* 

### 5. Set Up Environment Variables

For security and portability, API keys and configuration paths should not be hardcoded.

**A. Create a `.env` file** in the root directory of the project with the following content:

-   Replace `sk-xxxxxxxx...` with your actual LLM API key.
-   Replace `/path/to/your/videos` with the absolute path to the directory where your original video files are stored.

**B. Update `app.py` to use these variables.**


### 6. Configure Video URLs

The `video_url_map.json` file maps video filenames to their full URLs (e.g., on a SharePoint or cloud storage). Make sure this file is present and correctly populated with your video data.

The application will be available at http://127.0.0.1:5000. Open this URL in your web browser to start searching.


## Running the Application

Once the setup is complete, you can run the Flask application.


# Publications
1. Zhandi Liu, Mirna Becevic, Amy Braddock, Mihai Popescu, Eduardo J. Simoes, and Praveen Rao - **An Empirical Evaluation of Deep Learning Techniques for Clinical Video Retrieval**. In Proc. of 8th IEEE International Conference on Multimedia Information Processing and Retrieval (MIPR 2025), 8 pages, San Jose, CA. [[PDF]](https://drive.google.com/file/d/1twxy8q__LeYMgOGfMS7bdjnMLEeHuds2/view)


# Acknowledgments
This work was supported by the National Institute of Diabetes And Digestive And Kidney Diseases of the National Institutes of Health under Award Number P30DK092950.
