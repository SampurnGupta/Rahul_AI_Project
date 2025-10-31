# Advanced PDF Chatbot with LangGraph

An intelligent PDF chatbot that uses Pinecone for vector search, OpenAI LLM for generating answers, and LangGraph for efficient workflow management. This enhanced version includes smart relevance checking, multiple user interfaces, and flexible file input methods.

## 🚀 Multiple Ways to Use

### 1. **Web API** - For Integration & Remote Access
```bash
python api.py
# Access at http://localhost:8000
# Supports both URL and file upload
```

### 2. **Command Line Interface** - For Power Users
```bash
python chat_cli.py --pdf "your_document.pdf"
# Interactive file selection available
```

### 3. **Graphical Interface** - For Easy Use
```bash
python chat_gui.py
# User-friendly GUI with drag-and-drop
```

### 4. **Python Script** - For Developers
```bash
python advanced_pdf_bot.py --pdf "document.pdf"
# Direct Python integration
```

## 📁 Flexible File Input

**No more hardcoded paths!** The chatbot now supports:
- ✅ **Command line arguments**: `--pdf "path/to/file.pdf"`
- ✅ **Interactive selection**: Choose from files in current directory
- ✅ **Manual path entry**: Enter any file path when prompted
- ✅ **File uploads**: Upload files directly via web API
- ✅ **URL processing**: Process PDFs from web URLs
- ✅ **Relative/absolute paths**: Works with any valid file path
- ✅ **Home directory**: Supports `~/Documents/file.pdf` syntax

## Enhanced Features

- 📄 **Page-wise PDF processing**: Extract and process text page by page
- 🧠 **Smart summary generation**: Document and page-level summaries for efficient filtering
- 🔍 **Relevance checking**: Pre-filter queries using document summary to save API costs
- 📖 **Page-aware context**: Enhanced context retrieval with page summaries and cross-page references
- 🔄 **LangGraph workflow**: Efficient query processing pipeline with conditional logic
- 💰 **API cost optimization**: Only process relevant queries to minimize OpenAI API usage
- 🎯 **Targeted responses**: Page-specific context for more accurate answers
- 🌐 **Web API**: RESTful API with file upload and URL processing
- 💻 **Multiple interfaces**: CLI, GUI, API, and direct Python usage

## 🚀 Quick Start

### Option 1: Easy CLI (Recommended for beginners)
```bash
python chat_cli.py
```
The script will find PDF files in your current directory and let you choose, or you can enter any file path.

### Option 2: Direct file processing
```bash
python chat_cli.py --pdf "path/to/your/document.pdf"
```

### Option 3: Graphical interface
```bash
python chat_gui.py
```

### Option 4: Web API
```bash
python api.py
# Then visit http://localhost:8000/docs for the API interface
```

### Option 5: Traditional Python script
```bash
python advanced_pdf_bot.py --pdf "your_file.pdf"
```

📖 **See [USER_GUIDE.md](USER_GUIDE.md) for complete usage instructions and examples.**

## How It Works

### 1. Document Processing Pipeline
- **Page Extraction**: PDF is processed page by page to maintain structure
- **Summary Generation**: Creates document-level and page-level summaries
- **Smart Chunking**: Text is chunked per page with cross-page context
- **Enhanced Metadata**: Each chunk includes page summaries and previous page context

### 2. Query Processing Workflow (LangGraph)
```
User Query → Relevance Check → Context Search → Answer Generation
     ↓              ↓              ↓              ↓
  Embedding    Compare with    Find relevant   Generate response
  Generation   Doc Summary     pages/chunks    using context
```

### 3. Intelligent Filtering
- Queries are first compared against the document summary
- Only relevant queries proceed to full context search
- Irrelevant queries receive immediate feedback without API costs

## Prerequisites

Before running the script, you need to obtain API keys for:

1. **OpenAI API**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Pinecone**: Visit [Pinecone Console](https://app.pinecone.io/)

## Installation

1. **Clone or download the project files**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Open the `.env` file
   - Replace the placeholder values with your actual API keys:
     ```
     OPENAI_API_KEY=your-actual-openai-api-key
     PINECONE_API_KEY=your-actual-pinecone-api-key
     PINECONE_ENVIRONMENT=your-pinecone-environment
     ```

## Usage

1. **Prepare your PDF document**:
   - Place your PDF file in the project directory or note its full path

2. **Run the chatbot**:
   ```bash
   python advanced_pdf_bot.py
   ```

3. **Follow the prompts**:
   - Enter the path to your PDF file when prompted
   - Wait for the document to be processed and indexed
   - Start asking questions about your document

4. **Chat with your PDF**:
   - Type questions about the document content
   - The bot will search for relevant information and provide answers
   - Type 'quit' or 'exit' to stop the chatbot

## Example Usage

```
🤖 Advanced PDF Chatbot with LangGraph
==================================================

Please enter the path to your PDF file: my_document.pdf

📊 Initializing Pinecone index...
Setting up Pinecone index: pdf-chatbot-index
Connected to index: pdf-chatbot-index

📄 Processing PDF document...
Extracting text from PDF: my_document.pdf
Processing 10 pages...
Page 1: 3 chunks
Page 2: 4 chunks
...
Generated document summary
Generated summary for page 1
...
Successfully upserted 156 vectors with summaries to Pinecone index
✅ Document processing completed!

💬 PDF Chatbot is ready! Ask questions about your document.
Features:
- ✅ Smart relevance checking (saves API costs)
- 📖 Page-aware context retrieval  
- 📝 Summary-enhanced responses
Type 'quit' or 'exit' to stop the chatbot.
--------------------------------------------------

Your question: What is the main topic of this document?

🔄 Processing your question...
🔍 Checking query relevance...
Query relevance score: 0.892
✅ Query is relevant to the document
🔍 Searching for relevant context...
Found relevant pages: [1, 2, 5]
✅ Found relevant context
🤔 Generating answer...
✅ Answer generated successfully

📝 Answer:
------------------------------
Based on the document content, the main topic appears to be...
------------------------------

Your question: How do I cook pasta?

🔄 Processing your question...
🔍 Checking query relevance...
Query relevance score: 0.234
❌ Query is not relevant to the document

📝 Answer:
------------------------------
I'm sorry, but your question doesn't seem to be related to the content of this document. Please ask questions about the topics covered in the PDF.
------------------------------
```

## Configuration

You can modify the following constants in the script:

- `EMBEDDING_MODEL`: OpenAI embedding model (default: "text-embedding-ada-002")
- `LLM_MODEL`: OpenAI language model (default: "gpt-4-turbo")
- `PINECONE_INDEX_NAME`: Name of your Pinecone index
- Relevance thresholds in `check_query_relevance()` function
- Chunk size and overlap in the `get_text_chunks_by_page()` function

## API Cost Optimization Features

### 1. **Relevance Pre-filtering**
- Queries are first compared against document summary
- Similarity threshold prevents irrelevant processing
- LLM fallback for edge cases

### 2. **Smart Context Retrieval**
- Page-level filtering before chunk search
- Reduced embedding calls for irrelevant content
- Targeted search within relevant pages only

### 3. **Enhanced Metadata**
- Page summaries reduce search space
- Cross-page context improves accuracy
- Minimal redundant API calls

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure your API keys are correct and active
   - Check that you have sufficient credits/quota

2. **PDF Reading Issues**:
   - Make sure the PDF contains readable text (not just images)
   - Try with a different PDF file

3. **Pinecone Connection Issues**:
   - Verify your Pinecone environment name
   - Check your internet connection

4. **Memory Issues**:
   - For very large PDFs, consider reducing chunk size
   - Process smaller sections at a time

### Getting Help

If you encounter issues:
1. Check the error messages for specific details
2. Verify all dependencies are installed correctly
3. Ensure API keys are properly configured
4. Try with a smaller test PDF first

## File Structure

```
project/
├── advanced_pdf_bot.py    # Main chatbot script
├── .env                   # Environment variables (API keys)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── your_document.pdf     # Your PDF file
```

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure and private
- Regularly rotate your API keys for security

## License

This project is for educational and personal use. Please respect the terms of service of OpenAI and Pinecone APIs.
