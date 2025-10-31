# PDF Chatbot - User Guide

An intelligent PDF chatbot that allows you to ask questions about your PDF documents using advanced AI and vector search technology.

## 🚀 Quick Start

The PDF Chatbot now supports multiple ways to process your documents:

### 1. Web API (Recommended for Integration)
- **URL Processing**: Send PDF URLs for processing
- **File Upload**: Upload PDF files directly
- **Bearer Token Authentication** for security

### 2. Command Line Interface
- Interactive CLI with file selection
- Command line arguments support
- Perfect for power users and scripting

### 3. Graphical User Interface
- Easy-to-use GUI with tkinter
- Drag-and-drop file selection
- Real-time chat interface

### 4. Python Script
- Direct Python integration
- Customizable for developers

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Required environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=gcp-starter  # or your Pinecone environment
```

## 🌐 Web API Usage

### Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

### Endpoints

#### 1. Process PDF from URL
```http
POST /hackrx/run
Content-Type: application/json
Authorization: Bearer your_api_key

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key findings?"
  ]
}
```

#### 2. Upload PDF File
```http
POST /upload
Content-Type: multipart/form-data
Authorization: Bearer your_api_key

file: [PDF file]
questions: ["What is this document about?", "What are the conclusions?"]
```

### Response Format
```json
{
  "answers": [
    "This document discusses...",
    "The key findings are..."
  ]
}
```

## 💻 Command Line Interface

### Interactive Mode
```bash
python chat_cli.py
```
- Automatically finds PDF files in current directory
- Interactive file selection
- Enhanced chat interface with commands

### Direct File Processing
```bash
python chat_cli.py --pdf "path/to/your/document.pdf"
```

### Command Line Options
```bash
python chat_cli.py --help
```

### CLI Commands During Chat
- `help` or `h` - Show help message
- `info` - Display document information
- `quit`, `exit`, or `q` - Exit the chatbot

## 🖥️ Graphical User Interface

### Start the GUI
```bash
python chat_gui.py
```

### GUI Features
- **File Browser**: Easy PDF file selection
- **Real-time Processing**: Visual progress indicators
- **Chat Interface**: Natural conversation flow
- **Multi-threading**: Non-blocking UI during processing

## 🐍 Python Script Usage

### Basic Usage
```bash
python advanced_pdf_bot.py --pdf "path/to/document.pdf"
```

### With Command Line Arguments
```bash
python advanced_pdf_bot.py --pdf "document.pdf" --force-reprocess
```

### Interactive Mode
```bash
python advanced_pdf_bot.py
```
The script will prompt you to enter a file path.

## 📂 File Input Methods

### 1. Absolute Paths
```bash
python chat_cli.py --pdf "C:\Users\Username\Documents\report.pdf"
python chat_cli.py --pdf "/home/user/documents/report.pdf"
```

### 2. Relative Paths
```bash
python chat_cli.py --pdf "./documents/report.pdf"
python chat_cli.py --pdf "report.pdf"  # File in current directory
```

### 3. User Home Directory
```bash
python chat_cli.py --pdf "~/Documents/report.pdf"
```

### 4. Interactive Selection
- Run without `--pdf` argument
- Choose from files in current directory
- Or enter path manually

## 🔧 Configuration Options

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Optional
PINECONE_ENVIRONMENT=gcp-starter
HACKRX_API_KEY=your_custom_api_key
ENVIRONMENT=hackrx  # or production, external_testing
PORT=8000  # For API server
```

### Force Reprocessing
If you've updated a PDF or want to regenerate embeddings:
```bash
python chat_cli.py --pdf "document.pdf" --force-reprocess
```

## 📱 Example Usage Scenarios

### 1. Research Paper Analysis
```bash
python chat_cli.py --pdf "research_paper.pdf"
# Ask: "What is the main hypothesis of this study?"
# Ask: "What are the experimental results?"
```

### 2. Legal Document Review
```bash
python chat_gui.py
# Load contract.pdf through GUI
# Ask: "What are the termination clauses?"
# Ask: "What are the payment terms?"
```

### 3. API Integration
```python
import requests

response = requests.post(
    "http://localhost:8000/hackrx/run",
    headers={"Authorization": "Bearer your_api_key"},
    json={
        "documents": "https://example.com/manual.pdf",
        "questions": ["How do I install this software?"]
    }
)
print(response.json()["answers"][0])
```

## 🚨 Troubleshooting

### Common Issues

#### 1. File Not Found
- Check file path is correct
- Use absolute paths if relative paths don't work
- Ensure PDF file exists and is readable

#### 2. API Key Errors
- Verify environment variables are set
- Check API key validity
- Ensure sufficient API credits

#### 3. PDF Processing Errors
- Ensure PDF contains readable text (not just images)
- Check file size (large files may take longer)
- Try with `--force-reprocess` flag

#### 4. Permission Errors
- Ensure read permissions on PDF file
- Check write permissions for temporary files
- Run with appropriate user permissions

### Debug Mode
Enable debug logging:
```bash
python chat_cli.py --pdf "document.pdf" --log-level DEBUG
```

## 🎯 Features

### ✅ Smart Features
- **Relevance Checking**: Only processes questions related to document content
- **Page-aware Context**: Provides specific page references in answers
- **Citation Support**: Shows sources for all answers
- **Persistent Storage**: Reuses processed documents to save time
- **Multi-format Support**: Handles various PDF types

### ✅ User Experience
- **Multiple Interfaces**: CLI, GUI, API, and Python script
- **Interactive Selection**: Choose from available PDF files
- **Progress Indicators**: Visual feedback during processing
- **Error Handling**: Comprehensive error messages and recovery

### ✅ Performance
- **Caching**: Avoids reprocessing same documents
- **Optimized Embeddings**: Uses efficient OpenAI models
- **Background Processing**: Non-blocking operations in GUI

## 📄 Supported File Types

- **PDF files** (.pdf extension required)
- **Text-based PDFs** (scanned images may not work well)
- **Multi-page documents**
- **Various sizes** (reasonable limits apply)

## 🔒 Security

- **Bearer Token Authentication** for API access
- **Environment-based Configuration** for different deployment scenarios
- **Input Validation** for all user inputs
- **Secure File Handling** with temporary file cleanup

## 🆘 Getting Help

1. **Built-in Help**: Type `help` in any CLI interface
2. **API Documentation**: Visit `/docs` endpoint when API is running
3. **Error Messages**: Read error messages carefully for specific guidance
4. **Log Files**: Check `pdf_chatbot.log` for detailed debugging information

---

## Quick Reference

| Interface | Command | Use Case |
|-----------|---------|----------|
| **API Server** | `python api.py` | Web integration, remote access |
| **CLI Interactive** | `python chat_cli.py` | Quick local usage |
| **CLI Direct** | `python chat_cli.py --pdf file.pdf` | Scripting, automation |
| **GUI** | `python chat_gui.py` | User-friendly interface |
| **Python Script** | `python advanced_pdf_bot.py --pdf file.pdf` | Development, customization |

Choose the interface that best fits your needs and start chatting with your PDFs! 🚀
