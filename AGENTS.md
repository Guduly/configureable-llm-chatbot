# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in this repository.

## Project Overview

This is a configurable RAG (Retrieval-Augmented Generation) chatbot built with Python and LangChain. The bot can be configured to work with different document domains (medical, legal, tech) and provides question-answering capabilities based on loaded documents.

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry
poetry install

# Activate virtual environment
poetry shell

# Alternative: Use Python venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # (if available)
```

### Running the Application
```bash
# Run the main chatbot application
python rag_chat.py

# Run with specific model
python rag_chat.py --model chatter
```

### Testing
```bash
# No formal test suite currently exists
# To test manually:
python rag_chat.py
# Load a domain and ask questions
```

### Code Quality
```bash
# No linting commands configured yet
# Recommended to add:
# poetry add black flake8 mypy
# black rag_chat.py
# flake8 rag_chat.py
# mypy rag_chat.py
```

## Code Style Guidelines

### Python Standards
- **Python Version**: >=3.10,<3.13
- **Code Style**: Follow PEP 8
- **Line Length**: 88 characters (Black standard)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings and docstrings

### Import Organization
```python
# Standard library imports first
import os
import shutil
from pathlib import Path

# Third-party imports next
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports (if any)
# from .utils import helper_function
```

### Type Hints
- Use type hints for function parameters and return values
- Import from `typing` module when needed
- Example: `def load_domain(self, domain_name: str, docs_folder: str) -> bool:`

### Naming Conventions
- **Classes**: PascalCase (e.g., `ConfigurableDocBot`)
- **Functions/Methods**: snake_case (e.g., `load_domain`, `_process_documents`)
- **Variables**: snake_case (e.g., `vectorstore`, `current_domain`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `CHUNK_SIZE`, `MAX_TOKENS`)
- **Private methods**: Prefix with underscore (e.g., `_load_documents`)

### Error Handling
- Use specific exception types
- Include meaningful error messages
- Log errors appropriately
- Example:
```python
try:
    result = self.qa_chain.invoke({"query": question})
except Exception as e:
    print(f"❌ Error processing question: {str(e)}")
    return f"Error processing question: {str(e)}"
```

### Documentation
- Use docstrings for all classes and public methods
- Follow Google-style or NumPy-style docstring format
- Include parameter descriptions and return types
- Example:
```python
def load_domain(self, domain_name: str, docs_folder: str) -> bool:
    """Load documents for a specific domain/customer.
    
    Args:
        domain_name: Name of the domain to load
        docs_folder: Path to the documents folder
        
    Returns:
        True if loading was successful, False otherwise
    """
```

### Logging and Output
- Use print statements with emoji indicators for user feedback
- Structure output with clear visual separators
- Include progress indicators for long operations
- Example:
```python
print("🚀 Initializing Configurable Document Bot...")
print("⏳ Loading embeddings model...")
print("✅ Embeddings model loaded!")
```

## Project Structure

```
configureable-llm-chatbot/
├── rag_chat.py              # Main application file
├── pyproject.toml          # Poetry configuration
├── README.md               # Project documentation
├── Documents/              # Document collections by domain
│   ├── Medical/
│   ├── Legal/
│   └── Tech/
├── chroma_db_*/            # Vector databases (auto-generated)
└── .venv/                  # Virtual environment
```

## Key Dependencies

- **langchain**: Core framework for LLM applications
- **langchain-ollama**: Ollama LLM integration
- **langchain-chroma**: Chroma vector database integration
- **langchain-huggingface**: HuggingFace embeddings
- **chromadb**: Vector database for document storage
- **sentence-transformers**: Text embedding models
- **pypdf**: PDF document processing

## Development Guidelines

### Adding New Features
1. Follow the existing class structure in `ConfigurableDocBot`
2. Use dependency injection for LLM and embeddings
3. Implement proper error handling and user feedback
4. Add configuration options to the constructor
5. Test with different document types and domains

### Working with Documents
- Support markdown (.md), text (.txt), and PDF (.pdf) files
- Use `RecursiveCharacterTextSplitter` for chunking
- Default chunk size: 1000 characters with 200 overlap
- Store metadata including source file paths

### Vector Database Management
- Each domain gets its own Chroma database: `./chroma_db_{domain_name}`
- Clean up old databases when loading new domains
- Use `all-MiniLM-L6-v2` embeddings model
- Persist databases to disk for faster reloads

### Prompt Engineering
- Use clear, structured prompt templates
- Include context and question placeholders
- Add specific instructions for answer formatting
- Handle cases where information is not available

## Configuration

### Model Configuration
- Default model: `chatter` (custom fine-tuned model)
- Can be changed via constructor parameter
- Models are served via Ollama

### Embedding Configuration
- Default: `all-MiniLM-L6-v2`
- Device: CPU (configurable)
- Model downloads automatically on first use

### Retrieval Configuration
- Search type: similarity
- Retrieved documents: 4 (top-k)
- Chunk overlap: 200 characters
- Chunk size: 1000 characters

## Common Tasks

### Adding a New Document Type
1. Add file extension to `_load_documents()` method
2. Implement parsing logic with proper error handling
3. Create Document object with metadata
4. Test with sample files

### Switching LLM Models
1. Update model name in constructor or main()
2. Ensure model is available in Ollama
3. Test with different questions

### Customizing Prompts
1. Modify `prompt_template` in `_setup_qa_chain()`
2. Test with various question types
3. Ensure instructions are clear and specific

## Security Considerations

- Do not commit sensitive documents to version control
- Use environment variables for API keys (if needed)
- Validate user inputs properly
- Handle file access errors gracefully

## Performance Notes

- First-time embedding model download: ~90MB
- Vector database creation can be time-consuming for large document sets
- Consider chunk size vs. retrieval quality trade-offs
- Monitor memory usage with large document collections