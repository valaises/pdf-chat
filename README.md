# Chat with PDF - Document Search Tools

A standalone server that allows users to upload PDF documents and after documents are parsed and processed, users are able to call `/v1/tools-execute`, passing a conversation with pending tool calls. Server answers tool calls, producing new Messages with Results: RAG Messages, Helper Messages, or Text Messages.


Please, visit http://HOSTNAME/docs to access the API documentation

Alternatively, RAW OpenAPI documentation can be accessed at http://HOSTNAME/v1/openapi.json
## Overview

This application provides a complete pipeline for:
1. Uploading PDF files OR putting PDF files into shared folder
2. Extracting text content from PDFs
3. Processing the extracted content, uploading to OpenAI's File and Vector Store APIs.
4. Providing an API interface to List and Execute Tools

## Project Structure

```
chat-with-pdf-poc/
├── .env.example           # Example structure of ENV variables
├── Dockerfile             # Docker configuration
├── docker-compose.yaml    # Docker Compose configuration
├── README.md              # Project documentation
├── pyproject.toml         # Python project configuration
└── src/                   # Source code
    ├── core/              # Core application components
    │   ├── app.py         # FastAPI application setup
    │   ├── args.py        # Command line argument parsing
    │   ├── globals.py     # Global constants and settings
    │   ├── logger.py      # Logging configuration
    │   ├── main.py        # Application entry point
    │   ├── server.py      # Server configuration
    │   ├── repositories/  # Data access layer
    │   │   ├── repo_abstract.py  # Abstract repository base class
    │   │   └── repo_files.py     # File metadata storage and retrieval
    │   ├── routers/       # API endpoints
    │   │   ├── router_base.py    # Base router functionality
    │   │   ├── router_files.py   # File management endpoints
    │   │   ├── router_mcpl.py    # Model Context Protocol endpoints
    │   │   └── schemas.py        # API request/response schemas
    │   ├── tools/         # Tool implementations for MCPL
    │   │   ├── tool_context.py   # Context management for tools
    │   │   ├── tool_list_files.py # File listing tool
    │   │   ├── tool_search_in_file.py # File search tool
    │   │   └── tools.py          # Tool registration and management
    │   └── workers/       # Background processing workers
    │       ├── w_abstract.py     # Abstract worker base class
    │       ├── w_extractor.py    # PDF text extraction worker
    │       ├── w_processor.py    # PDF Text processing worker -- uploads to OpenAI
    │       └── w_utils.py        # Utility functions for workers
    ├── coxit/
    │   └── extractor/     # PDF text extraction components
    │       └── ...
    └── openai_wrappers/   # OpenAI API wrappers
        ├── api_files.py   # OpeanAI's File API wrappers
        ├── api_vector_store.py # OpenAI's Vector store API wrappers
        ├── types.py       # Type definitions
        └── utils.py       # Utility functions
```
## Features

- **File Management**: Upload, list, and delete PDF files
- **Asynchronous Processing**: Background workers handle resource-intensive tasks
- **Vector Search**: Semantic search capabilities using OpenAI's vector stores
- **Stateful Processing**: Track processing status of documents from upload to completion

## Technical Details

### Architecture

The application follows a modular architecture with these key components:

1. **FastAPI Web Server**: Handles HTTP requests for file uploads, listing, and chat interactions
2. **Background Workers**: Additional Threads that process files without blocking the web server (CPU/IO intensive tasks)
3. **Repository Layer**: Abstracts database operations for file metadata
4. **PDF Extraction Library**: Custom approaches for extracting structured text from PDFs
5. **OpenAI Integration**: Wrappers around OpenAI's API for vector stores and file uploads

### Storage

- **SQLite Database**: Stores file metadata including:
  - Original and hashed filenames
  - User ID
  - Creation timestamp
  - Processing status
  - Vector store ID
  
- **File System Storage**:
  - Uploaded PDFs stored with hashed filenames
  - Extracted text stored in JSONL format
  - Visualization of extracted paragraphs (optional)
  
- **OpenAI Vector Stores**: 
  - Semantic search capabilities using OpenAI's embeddings
  - Enables natural language querying of document content

### Processing Pipeline

1. **Upload Phase**:
   - User uploads a PDF file via the `/v1/file-upload` endpoint OR
     File is put into shared folder and `v1/file-create` is called to mark file to process.
   - File is saved with a hashed filename
   - Database record of a file is created with empty processing status

2. **Extraction Phase**:
   - The extractor worker monitors for new files
   - When a new file is detected, text is extracted from the PDF
   - Extraction uses the coxit library to parse paragraphs and sections, assign highlight coordinates
   - Status is updated to `"extracted"` when complete

3. **Processing Phase**:
   - The processor worker monitors for files with `"extracted"` status
   - Status is updated to `"processing"` during this phase
   - A vector store is created if it does not exist
   - Paragraphs are uploaded as files into OpenAI's File API
   - Those 'files' are assigned to created vector stored
   - Status is updated to `"complete"` when finished
   - Any orphaned files are cleaned up from OpenAI's File and Vector Store APIs.

4. **Query Phase**:
   - User calls tools-execute with given messages in OpenAI format
   - IF messages (after latest user message) contain unanswered tool calls, tools are executed: list_documents or search_in_doc
   - After tools are validated, then executed, returning tool answer messages in output

### Status Tracking

Files progress through these statuses:
- Empty status: Newly uploaded, awaiting processing
- `"extracted"`: Text has been extracted from the PDF
- `"processing"`: Currently being processed by the processor worker
- `"incomplete"`: Processing was interrupted and needs to be resumed
- `"complete"`: Fully processed
- `"error: [message]"`: An error occurred during processing

## Getting Started

### Installation

1. Clone the repository
```bash
git clone [repository-url]
cd chat-with-pdf-poc
```

#### Copy the example environment file and edit it
```bash
cp .env.example .env
```
#### Build and run with Docker Compose
```bash
docker-compose up --build -d
```
