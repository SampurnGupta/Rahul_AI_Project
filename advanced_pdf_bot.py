"""
Advanced Document Chatbot: Retrieval-Augmented Generation with semantic search and explainable decisions.

Capabilities added for HackRx requirements:
- Ingest PDFs and DOCX (email/EML can be converted to text upstream)
- Chunking per page/section; summary vectors to guide retrieval
- Vector search via Pinecone (pluggable backend; FAISS stub present)
- Clause extraction and matching for policy/contract questions
- Explainable JSON outputs with rationale and traceability

Requirements:
- Environment variables in .env
- pip install -r requirements.txt
"""

import os
import re
import logging
import json
import time
from typing import List, Any, Dict, TypedDict, Tuple, Optional
import pypdf
try:
    from docx import Document as DocxDocument  # type: ignore[reportMissingImports]
except Exception:
    DocxDocument = None  # Optional dependency
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


def setup_logging(log_level: str = "INFO"):
    """
    Set up logging configuration with Unicode support for Windows.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import sys
    
    # Create a custom stream handler for Windows console
    class UnicodeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                # Replace Unicode emojis with simple text for Windows compatibility
                msg = msg.replace('🔍', '[SEARCH]')
                msg = msg.replace('✅', '[SUCCESS]')
                msg = msg.replace('🤔', '[THINKING]')
                msg = msg.replace('❌', '[ERROR]')
                msg = msg.replace('⚠️', '[WARNING]')
                
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except UnicodeEncodeError:
                # Fallback: encode to ASCII, ignoring problematic characters
                try:
                    msg_ascii = msg.encode('ascii', 'ignore').decode('ascii')
                    stream.write(msg_ascii + self.terminator)
                    self.flush()
                except Exception:
                    pass  # Silent fail to prevent logging loops
    
    # Set up logging with Unicode-safe handlers
    handlers = [
        UnicodeStreamHandler(sys.stdout),
        logging.FileHandler('pdf_chatbot.log', encoding='utf-8')
    ]
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Configuration - fetch API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")  # Default environment
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pinecone").lower()  # pinecone|faiss (faiss stub)

# Validate that all required environment variables are set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if VECTOR_BACKEND == "pinecone" and not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables for Pinecone backend")

# Initialize clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY) if VECTOR_BACKEND == "pinecone" else None

# Global constants
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
PINECONE_INDEX_NAME = "pdf-chatbot-index"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

# Speed/Token optimization flags
ENABLE_DOC_SUMMARY = os.getenv("ENABLE_DOC_SUMMARY", "true").lower() == "true"
ENABLE_PAGE_SUMMARIES = os.getenv("ENABLE_PAGE_SUMMARIES", "false").lower() == "true"
ENABLE_RELEVANCE_LLM_FALLBACK = os.getenv("ENABLE_RELEVANCE_LLM_FALLBACK", "false").lower() == "true"
ENABLE_CLAUSE_MATCHING = os.getenv("ENABLE_CLAUSE_MATCHING", "false").lower() == "true"
INCLUDE_SUMMARIES_IN_CONTEXT = os.getenv("INCLUDE_SUMMARIES_IN_CONTEXT", "false").lower() == "true"

PAGE_SUMMARY_TOP_K = int(os.getenv("PAGE_SUMMARY_TOP_K", "2"))
CHUNK_TOP_K = int(os.getenv("CHUNK_TOP_K", "6"))
CONTEXT_CHAR_BUDGET = int(os.getenv("CONTEXT_CHAR_BUDGET", "3000"))

# ---------------------------
# Utilities for DOC ingestion
# ---------------------------

def extract_text_from_docx(docx_path: str) -> Dict[int, str]:
    """Extract text from DOCX into pseudo-pages (paragraph blocks)."""
    if DocxDocument is None:
        raise ImportError("python-docx is not installed. Please install it to process DOCX files.")
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")
    doc = DocxDocument(docx_path)
    # Group paragraphs into blocks of ~800-1200 chars to act like pages
    blocks: Dict[int, str] = {}
    current: List[str] = []
    current_len = 0
    page_no = 1
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        current.append(text)
        current_len += len(text) + 1
        if current_len >= 1000:
            blocks[page_no] = "\n".join(current)
            page_no += 1
            current, current_len = [], 0
    if current:
        blocks[page_no] = "\n".join(current)
    if not blocks:
        blocks[1] = ""
    return blocks

# LangGraph State Definition
class ProcessingState(TypedDict):
    query: str
    pdf_path: str
    doc_id: str
    document_summary: str
    page_summaries: List[str]
    relevant_pages: List[int]
    context: str
    answer: str
    should_process: bool
    error: str


def extract_text_from_pdf(pdf_path: str) -> Dict[int, str]:
    """
    Extract text from a PDF file page by page using pypdf library.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Dict[int, str]: Dictionary with page numbers as keys and page text as values
        
    Raises:
        FileNotFoundError: If the PDF file is not found
        Exception: For other PDF processing errors
    """
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pages_text = {}
        print(f"Extracting text from PDF: {pdf_path}")

        # Open and read the PDF file
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"Processing {total_pages} pages...")

            # Extract text from each page separately
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                pages_text[page_num] = page_text
                print(f"Processed page {page_num}/{total_pages}")

        print(f"Successfully extracted text from {len(pages_text)} pages")
        return pages_text

    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        raise
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise


def get_text_chunks_by_page(pages_text: Dict[int, str]) -> Dict[int, List[str]]:
    """
    Split text into chunks page by page using RecursiveCharacterTextSplitter.
    
    Args:
        pages_text (Dict[int, str]): Dictionary with page numbers and their text
        
    Returns:
        Dict[int, List[str]]: Dictionary with page numbers and their chunks
    """
    print("Splitting text into chunks by page...")
    
    # Initialize the text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    page_chunks = {}
    total_chunks = 0
    
    for page_num, text in pages_text.items():
        if text.strip():  # Only process pages with content
            chunks = text_splitter.split_text(text)
            page_chunks[page_num] = chunks
            total_chunks += len(chunks)
            print(f"Page {page_num}: {len(chunks)} chunks")
    
    print(f"Created {total_chunks} total chunks across {len(page_chunks)} pages")
    return page_chunks


def generate_summary(text: str, summary_type: str = "document") -> str:
    """
    Generate a summary of the given text using OpenAI LLM.
    
    Args:
        text (str): Text to summarize
        summary_type (str): Type of summary ("document", "page", or "chunk")
        
    Returns:
        str: Generated summary
    """
    try:
        if summary_type == "document":
            prompt = f"""Please provide a comprehensive summary of this document. Include:
1. Main topics and themes
2. Key concepts and ideas
3. Important entities (people, places, organizations)
4. Overall purpose and scope

Document text:
{text[:4000]}"""  # Limit to avoid token limits
        
        elif summary_type == "page":
            prompt = f"""Please provide a concise summary of this page content. Include:
1. Main topics discussed
2. Key points and concepts
3. Important details

Page text:
{text}"""
        
        else:  # chunk
            prompt = f"""Please provide a brief summary of this text chunk focusing on:
1. Main topic
2. Key information
3. Context

Text chunk:
{text}"""

        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that creates concise, informative summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return f"Summary generation failed: {str(e)}"


def setup_pinecone_index(index_name: str, dimension: int, force_recreate: bool = False):
    """
    Set up Pinecone index - create if it doesn't exist, or connect to existing one.
    
    Args:
        index_name (str): Name of the Pinecone index
        dimension (int): Dimension of the embeddings (1536 for text-embedding-3-small)
        force_recreate (bool): If True, delete and recreate the index even if it exists
        
    Returns:
        tuple: (pinecone.Index, bool) - The Pinecone index object and whether it was newly created
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Setting up Pinecone index: {index_name}")
    
    # List existing indexes
    existing_indexes = [index.name for index in pinecone_client.list_indexes()]
    
    # Check if index exists
    if index_name in existing_indexes:
        if force_recreate:
            logger.info(f"Force recreating index {index_name}")
            try:
                pinecone_client.delete_index(index_name)
                logger.info(f"Index {index_name} deleted successfully")
            except Exception as e:
                logger.error(f"Error deleting existing index: {e}")
                raise
        else:
            # Try to connect to existing index
            try:
                index = pinecone_client.Index(index_name)
                # Test the connection and check stats
                stats = index.describe_index_stats()
                logger.info(f"Connected to existing index {index_name} with {stats.total_vector_count} vectors")
                return index, False
            except Exception as e:
                logger.warning(f"Error connecting to existing index: {e}")
                logger.info("Will recreate the index...")
                try:
                    pinecone_client.delete_index(index_name)
                    logger.info(f"Index {index_name} deleted successfully")
                except Exception as delete_error:
                    logger.error(f"Error deleting corrupted index: {delete_error}")
                    raise
    
    # Create new index
    logger.info(f"Creating new Pinecone index: {index_name}")
    try:
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        logger.info(f"Index {index_name} created successfully")
        
        # Connect to the new index
        index = pinecone_client.Index(index_name)
        logger.info(f"Connected to new index: {index_name}")
        return index, True
        
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise


def check_document_exists(index, doc_id: str) -> bool:
    """
    Check if a document already exists in the Pinecone index.
    
    Args:
        index: Pinecone index object
        doc_id (str): Document ID to check
        
    Returns:
        bool: True if document exists, False otherwise
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Query for document summary to check if document exists
        results = index.query(
            vector=[0.0] * EMBEDDING_DIMENSION,  # Dummy vector
            top_k=1,
            include_metadata=True,
            filter={"type": "document_summary", "doc_id": doc_id}
        )
        
        exists = len(results.matches) > 0
        if exists:
            logger.info(f"Document {doc_id} already exists in index")
        else:
            logger.info(f"Document {doc_id} not found in index")
            
        return exists
        
    except Exception as e:
        logger.warning(f"Error checking document existence: {e}")
        return False


def embed_and_upsert_with_summaries(index, pages_text: Dict[int, str], page_chunks: Dict[int, List[str]], doc_id: str):
    """
    Generate embeddings for text chunks and summaries, then upsert them into Pinecone index.

    Args:
        index: Pinecone index object
        pages_text (Dict[int, str]): Dictionary with page numbers and their text
        page_chunks (Dict[int, List[str]]): Dictionary with page numbers and their chunks
        doc_id (str): Unique identifier for the document
    """
    print("Generating embeddings and upserting with summaries...")

    try:
        doc_summary = ""
        if ENABLE_DOC_SUMMARY:
            # Generate and upsert document summary
            full_text = "\n".join(pages_text.values())
            doc_summary = generate_summary(full_text, "document")
            print("Generated document summary")

            doc_summary_embedding = openai_client.embeddings.create(
                input=[doc_summary],
                model=EMBEDDING_MODEL,
            ).data[0].embedding

            index.upsert(
                vectors=[
                    {
                        "id": f"{doc_id}-summary",
                        "values": doc_summary_embedding,
                        "metadata": {
                            "type": "document_summary",
                            "text": doc_summary,
                            "doc_id": doc_id,
                        },
                    }
                ]
            )

        # Process each page
        vectors_to_upsert: List[Dict[str, Any]] = []
        page_summaries: Dict[int, str] = {}

        for page_num in sorted(page_chunks.keys()):
            page_text = pages_text.get(page_num, "")
            chunks = page_chunks.get(page_num, [])

            if ENABLE_PAGE_SUMMARIES:
                # Generate page summary and upsert
                page_summary = generate_summary(page_text, "page")
                page_summaries[page_num] = page_summary
                print(f"Generated summary for page {page_num}")

                page_summary_embedding = openai_client.embeddings.create(
                    input=[page_summary],
                    model=EMBEDDING_MODEL,
                ).data[0].embedding

                vectors_to_upsert.append(
                    {
                        "id": f"{doc_id}-page-{page_num}-summary",
                        "values": page_summary_embedding,
                        "metadata": {
                            "type": "page_summary",
                            "text": page_summary,
                            "doc_id": doc_id,
                            "page_number": page_num,
                        },
                    }
                )

            # Generate embeddings for chunks
            if chunks:
                chunk_embeddings = openai_client.embeddings.create(
                    input=chunks,
                    model=EMBEDDING_MODEL,
                )

                # Create vectors for each chunk
                for chunk_idx, (chunk, embedding) in enumerate(
                    zip(chunks, chunk_embeddings.data)
                ):
                    # Include previous page summary as context if available
                    prev_summary = page_summaries.get(page_num - 1, "") if page_num > 1 else ""

                    meta = {
                        "type": "chunk",
                        "text": chunk,
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                    }
                    # Optionally include summaries in metadata (not needed for retrieval)
                    if ENABLE_PAGE_SUMMARIES:
                        meta["page_summary"] = page_summaries.get(page_num, "")
                    vectors_to_upsert.append(
                        {
                            "id": f"{doc_id}-page-{page_num}-chunk-{chunk_idx}",
                            "values": embedding.embedding,
                            "metadata": meta,
                        }
                    )

        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            index.upsert(vectors=batch)
            print(
                f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1)//batch_size}"
            )

        print(
            f"Successfully upserted {len(vectors_to_upsert)} vectors with summaries to Pinecone index"
        )
        return doc_summary, page_summaries

    except Exception as e:
        print(f"Error during embedding and upsert: {str(e)}")
        raise


def check_query_relevance(index, query: str, doc_id: str) -> tuple[bool, str]:
    """
    Check if the user query is relevant to the document by comparing with document summary.
    
    Args:
        index: Pinecone index object
        query (str): User's query
        doc_id (str): Document ID
        
    Returns:
        tuple[bool, str]: (is_relevant, document_summary)
    """
    try:
        # If document summaries are disabled, skip relevance filtering
        if not ENABLE_DOC_SUMMARY:
            return True, ""
        # Generate embedding for the query
        query_embedding = openai_client.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        ).data[0].embedding
        
        # Search for document summary
        search_results = index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
            filter={"type": "document_summary", "doc_id": doc_id}
        )
        
        if not search_results.matches:
            return False, "No document summary found"
        
        # Get similarity score
        similarity_score = search_results.matches[0].score
        document_summary = search_results.matches[0].metadata.get('text', '')
        
        # Threshold for relevance (can be adjusted)
        relevance_threshold = 0.2  # Lowered from 0.7 to 0.2 for better sensitivity
        
        print(f"Query relevance score: {similarity_score:.3f}")
        
        if similarity_score >= relevance_threshold:
            return True, document_summary
        else:
            if not ENABLE_RELEVANCE_LLM_FALLBACK:
                return False, document_summary
            # Use LLM to make final decision
            relevance_prompt = f"""
            Document Summary: {document_summary}
            
            User Query: {query}
            
            Is this query relevant to the document? Consider if the query could be answered using information from this document.
            Answer only 'YES' or 'NO' followed by a brief explanation.
            """
            
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines query relevance."},
                    {"role": "user", "content": relevance_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            answer = response.choices[0].message.content.strip().upper()
            is_relevant = answer.startswith('YES')
            
            return is_relevant, document_summary
            
    except Exception as e:
        print(f"Error checking query relevance: {str(e)}")
    return True, ""  # Default to processing if error occurs


def get_relevant_context_enhanced(index, query: str, doc_id: str, top_k: int = CHUNK_TOP_K) -> str:
    """
    Perform enhanced semantic search to get relevant context for the query.
    
    Args:
        index: Pinecone index object
        query (str): User's query
        doc_id (str): Document ID
        top_k (int): Number of top results to retrieve
        
    Returns:
        str: Combined context from retrieved chunks
    """
    try:
        # Generate embedding for the query
        query_embedding = openai_client.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        ).data[0].embedding
        
        relevant_pages = []
        # First, search for relevant page summaries if enabled
        if ENABLE_PAGE_SUMMARIES:
            page_summary_results = index.query(
                vector=query_embedding,
                top_k=PAGE_SUMMARY_TOP_K,
                include_metadata=True,
                filter={"type": "page_summary", "doc_id": doc_id}
            )

            if page_summary_results.matches:
                for match in page_summary_results.matches:
                    if match.score > 0.2:  # Lowered threshold for page summaries
                        page_num = match.metadata.get('page_number')
                        if page_num:
                            relevant_pages.append(page_num)
        
        # If we have relevant pages, search for chunks within those pages
        if relevant_pages:
            print(f"Found relevant pages: {relevant_pages}")
            context_chunks = []
            
            for page_num in relevant_pages:
                # Get chunks from this specific page
                chunk_results = index.query(
                    vector=query_embedding,
                    top_k=CHUNK_TOP_K,
                    include_metadata=True,
                    filter={"type": "chunk", "doc_id": doc_id, "page_number": page_num}
                )
                
                for match in chunk_results.matches:
                    if match.score > 0.2:  # Lowered relevance threshold for chunks
                        chunk_text = match.metadata.get('text', '')
                        page_summary = match.metadata.get('page_summary', '')

                        # Create enhanced context; include summaries if configured
                        if INCLUDE_SUMMARIES_IN_CONTEXT and page_summary:
                            enhanced_chunk = f"""
                            [Page {page_num} Context]
                            Page Summary: {page_summary}

                            Content: {chunk_text}
                            """
                        else:
                            enhanced_chunk = f"[Page {page_num}] {chunk_text}"
                        context_chunks.append(enhanced_chunk)
            
            if context_chunks:
                combined = "\n\n".join(context_chunks[:top_k])
                # Enforce character budget
                if len(combined) > CONTEXT_CHAR_BUDGET:
                    combined = combined[:CONTEXT_CHAR_BUDGET]
                return combined
        
        # Fallback: general search across all chunks
        print("Performing general search across all chunks...")
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"type": "chunk", "doc_id": doc_id}
        )
        
        context_chunks = []
        for match in search_results.matches:
            if 'text' in match.metadata and match.score > 0.2:  # Lowered threshold
                chunk_text = match.metadata['text']
                page_num = match.metadata.get('page_number', 'Unknown')
                if INCLUDE_SUMMARIES_IN_CONTEXT and ENABLE_PAGE_SUMMARIES:
                    page_summary = match.metadata.get('page_summary', '')
                    if page_summary:
                        context_chunks.append(f"""
                        [Page {page_num} Context]
                        Page Summary: {page_summary}

                        Content: {chunk_text}
                        """)
                    else:
                        context_chunks.append(f"[Page {page_num}] {chunk_text}")
                else:
                    context_chunks.append(f"[Page {page_num}] {chunk_text}")

        combined = "\n\n".join(context_chunks)
        if len(combined) > CONTEXT_CHAR_BUDGET:
            combined = combined[:CONTEXT_CHAR_BUDGET]
        return combined
        
    except Exception as e:
        print(f"Error during enhanced semantic search: {str(e)}")
        return ""


def get_llm_answer(context: str, query: str) -> str:
    """
    Generate a direct, concise answer using OpenAI LLM based on the provided context.
    
    Args:
        context (str): Retrieved context from semantic search
        query (str): User's original query
        
    Returns:
        str: A direct string answer to the question.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Simplified prompt for direct answers, optimized for token efficiency
    prompt = f"""You are an expert assistant for policy documents. Answer the user's question based *only* on the provided context.
If the answer is not in the context, state that you cannot answer based on the provided information.
Be concise and directly answer the question.

Context:
---
{context}
---
Question: {query}

Answer:"""

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that provides direct answers based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Set to 0 for fact-based, deterministic answers
            max_tokens=300  # Reduced max_tokens for concise answers
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated answer for query: {query[:50]}...")
        return answer

    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        return "Sorry, I encountered an error while generating the response."


# -----------------------------
# Clause extraction and matching
# -----------------------------

def extract_candidate_clauses(pages_text: Dict[int, str]) -> List[Dict[str, Any]]:
    """Heuristic clause segmentation: split by headings and numbering."""
    clauses: List[Dict[str, Any]] = []
    heading_re = re.compile(r"^(\d+\.|[A-Z][\w\s]{1,40}:?)\s", re.MULTILINE)
    for page, text in pages_text.items():
        if not text:
            continue
        # naive split by double newline boundaries
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        for part in parts:
            title = None
            m = heading_re.match(part)
            if m:
                title = m.group(0).strip().rstrip(":")
            snippet = part[:400]
            clauses.append({"page": page, "title": title or "Clause", "text": part, "snippet": snippet})
    return clauses

def rank_clauses_by_similarity(clauses: List[Dict[str, Any]], query: str) -> List[Tuple[Dict[str, Any], float]]:
    """Embed query and clause snippets; return ranked matches with similarity scores."""
    if not clauses:
        return []
    # Prepare inputs
    inputs = [query] + [c["snippet"] for c in clauses]
    embs = openai_client.embeddings.create(input=inputs, model=EMBEDDING_MODEL).data
    q = embs[0].embedding
    def cosine(a, b):
        import math
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        return dot / (na*nb + 1e-8)
    scored: List[Tuple[Dict[str, Any], float]] = []
    for idx, c in enumerate(clauses, start=1):
        s = cosine(q, embs[idx].embedding)
        scored.append((c, float(s)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def check_relevance_node(state: ProcessingState) -> ProcessingState:
    """Check if the query is relevant to the document."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Checking query relevance...")
    
    try:
        index = pinecone_client.Index(PINECONE_INDEX_NAME)
        is_relevant, doc_summary = check_query_relevance(index, state["query"], state["doc_id"])
        
        state["should_process"] = is_relevant
        state["document_summary"] = doc_summary
        
        if is_relevant:
            logger.info("✅ Query is relevant to the document")
        else:
            logger.info("❌ Query is not relevant to the document")
            
    except Exception as e:
        logger.error(f"Error in relevance checking: {e}")
        state["error"] = str(e)
        state["should_process"] = False
        
    return state


def search_context_node(state: ProcessingState) -> ProcessingState:
    """Search for relevant context if query is relevant."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not state["should_process"]:
        return state
        
    logger.info("🔍 Searching for relevant context...")
    
    try:
        index = pinecone_client.Index(PINECONE_INDEX_NAME)
        context = get_relevant_context_enhanced(index, state["query"], state["doc_id"])
        state["context"] = context
        
        if context:
            logger.info("✅ Found relevant context")
        else:
            logger.warning("❌ No relevant context found")
            
    except Exception as e:
        logger.error(f"Error in context search: {e}")
        state["error"] = str(e)
        
    return state


def generate_answer_node(state: ProcessingState) -> ProcessingState:
    """Generate answer using LLM."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not state["should_process"] or not state["context"]:
        if not state["should_process"]:
            answer = "I'm sorry, but your question doesn't seem to be related to the content of this document. Please ask questions about the topics covered in the PDF."
        else:
            answer = "I couldn't find relevant information in the document to answer your question."
        state["answer"] = answer
        return state
        
    logger.info("🤔 Generating answer...")
    
    try:
        answer = get_llm_answer(state["context"], state["query"])
        state["answer"] = answer
        logger.info("✅ Answer generated successfully")
        
    except Exception as e:
        logger.error(f"Error in generate_answer_node: {e}")
        state["error"] = str(e)
        state["answer"] = f"Sorry, I encountered an error while generating the response: {str(e)}"
        
    return state


def should_process_query(state: ProcessingState) -> str:
    """Conditional edge function to determine if query should be processed."""
    if state["should_process"]:
        return "search_context"
    else:
        return "generate_answer"


# Create LangGraph workflow
def create_workflow():
    """Create the LangGraph workflow for processing queries."""
    workflow = StateGraph(ProcessingState)
    
    # Add nodes
    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("search_context", search_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # Add edges
    workflow.set_entry_point("check_relevance")
    workflow.add_conditional_edges(
        "check_relevance",
        should_process_query,
        {
            "search_context": "search_context",
            "generate_answer": "generate_answer"
        }
    )
    workflow.add_edge("search_context", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # Compile workflow
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


class PDFChatbot:
    """Enhanced PDF Chatbot with LangGraph workflow."""
    
    def __init__(self, pdf_path: str, force_reprocess: bool = False):
        self.pdf_path = pdf_path
        self.doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        self.index = None
        self.workflow = create_workflow()
        self.document_summary = ""
        self.force_reprocess = force_reprocess
        self.logger = logging.getLogger(__name__)
        self.is_newly_created = False
        self._pages_cache: Optional[Dict[int, str]] = None
        
    def setup_index(self):
        """Set up Pinecone index."""
        self.logger.info("📊 Initializing vector index...")
        if VECTOR_BACKEND == "pinecone":
            self.index, self.is_newly_created = setup_pinecone_index(
                PINECONE_INDEX_NAME,
                EMBEDDING_DIMENSION,
                force_recreate=self.force_reprocess
            )
        else:
            raise ValueError("FAISS backend not implemented in this build. Set VECTOR_BACKEND=pinecone.")
        
    def process_document(self):
        """Process the PDF document and store in vector database."""
        self.logger.info("📄 Processing document...")
        
        # Check if document already exists (unless force reprocessing)
        if not self.force_reprocess and not self.is_newly_created:
            if check_document_exists(self.index, self.doc_id):
                self.logger.info(f"✅ Document {self.doc_id} already exists in index, skipping processing")
                # Get existing document summary
                try:
                    results = self.index.query(
                        vector=[0.0] * EMBEDDING_DIMENSION,
                        top_k=1,
                        include_metadata=True,
                        filter={"type": "document_summary", "doc_id": self.doc_id}
                    )
                    if results.matches:
                        self.document_summary = results.matches[0].metadata.get('text', '')
                        self.logger.info("Retrieved existing document summary")
                except Exception as e:
                    self.logger.warning(f"Could not retrieve existing summary: {e}")
                return
        
        # Extract text per format
        if self.pdf_path.lower().endswith(".pdf"):
            pages_text = extract_text_from_pdf(self.pdf_path)
        elif self.pdf_path.lower().endswith(".docx"):
            pages_text = extract_text_from_docx(self.pdf_path)
        else:
            raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")
        self._pages_cache = pages_text
        
        if not any(text.strip() for text in pages_text.values()):
            raise ValueError("No text was extracted from the PDF. Please check if the PDF contains readable text.")
        
        # Create chunks by page
        page_chunks = get_text_chunks_by_page(pages_text)
        
        # Generate embeddings and upsert with summaries
        self.logger.info(f"🔄 Processing document with ID: {self.doc_id}")
        doc_summary, page_summaries = embed_and_upsert_with_summaries(
            self.index, pages_text, page_chunks, self.doc_id
        )
        
        self.document_summary = doc_summary
        self.logger.info("✅ Document processing completed!")
        
    def ask_question(self, query: str) -> str:
        """Ask a question using the LangGraph workflow and return a direct string answer."""
        initial_state = ProcessingState(
            query=query,
            pdf_path=self.pdf_path,
            doc_id=self.doc_id,
            document_summary="",
            page_summaries=[],
            relevant_pages=[],
            context="",
            answer="",
            should_process=False,
            error=""
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": f"pdf_chat_session_{self.doc_id}"}}
        result = self.workflow.invoke(initial_state, config)
        
        # The answer is now a direct string
        return result.get("answer", "No answer could be generated.")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    print("🤖 Advanced PDF Chatbot with LangGraph")
    print("=" * 50)
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Advanced PDF Chatbot with LangGraph")
    parser.add_argument(
        "--pdf", 
        type=str, 
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "--force-reprocess", 
        action="store_true", 
        help="Force reprocessing of the document even if it exists in the index"
    )
    
    args = parser.parse_args()
    
    # Get PDF file path
    pdf_path = args.pdf
    
    if not pdf_path:
        print("📁 No PDF file specified. Please provide a PDF file.")
        print("\nOptions:")
        print("1. Use command line: python advanced_pdf_bot.py --pdf \"path/to/your/file.pdf\"")
        print("2. Enter path interactively")
        print()
        
        # Interactive file path input
        while True:
            pdf_path = input("Enter the path to your PDF file: ").strip()
            
            if not pdf_path:
                print("❌ Please enter a valid file path.")
                continue
                
            # Remove quotes if user added them
            pdf_path = pdf_path.strip('"').strip("'")
            
            # Check if file exists
            if os.path.exists(pdf_path):
                if pdf_path.lower().endswith('.pdf'):
                    break
                else:
                    print("❌ File must be a PDF (.pdf extension required).")
                    continue
            else:
                print(f"❌ File not found: {pdf_path}")
                print("Please check the file path and try again.")
                continue
    
    print(f"Processing PDF: {pdf_path}")
    logger.info(f"Starting PDF chatbot with file: {pdf_path}")
    
    # Final check if file exists
    if not os.path.exists(pdf_path):
        error_msg = f"❌ Error: PDF file not found at {pdf_path}"
        print(error_msg)
        logger.error(error_msg)
        print("Please check the file path and try again.")
        sys.exit(1)
    
    try:
        # Initialize the chatbot
        force_reprocess = args.force_reprocess if args.force_reprocess else False
        chatbot = PDFChatbot(pdf_path, force_reprocess=force_reprocess)
        
        # Setup index and process document
        chatbot.setup_index()
        chatbot.process_document()
        
        # Interactive Q&A loop
        print("\n💬 PDF Chatbot is ready! Ask questions about your document.")
        print("Features:")
        print("- ✅ Smart relevance checking (saves API costs)")
        print("- ✅ Page-aware context retrieval")
        print("- ✅ Summary-enhanced responses with citations")
        print("- ✅ Persistent vector storage")
        print("- ✅ Enhanced error handling and logging")
        print("\nCommands:")
        print("- Type your question to get an answer")
        print("- Type 'quit', 'exit', or 'q' to stop the chatbot")
        print("- Type 'help' for more options")
        print("-" * 50)
        
        while True:
            # Get user query
            user_query = input("\nYour question: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the PDF Chatbot!")
                logger.info("User ended session")
                break
            
            # Help command
            if user_query.lower() in ['help', 'h']:
                print("\n📖 Available Commands:")
                print("- Ask any question about the PDF document")
                print("- 'quit', 'exit', 'q' - Exit the chatbot")
                print("- 'help', 'h' - Show this help message")
                print("- 'info' - Show document information")
                print("- 'stats' - Show session statistics")
                continue
            
            # Info command
            if user_query.lower() == 'info':
                print(f"\n📄 Document Information:")
                print(f"- File: {os.path.basename(pdf_path)}")
                print(f"- Full path: {pdf_path}")
                print(f"- Document ID: {chatbot.doc_id}")
                if chatbot.document_summary:
                    print(f"- Summary: {chatbot.document_summary[:200]}...")
                continue
            
            # Stats command
            if user_query.lower() == 'stats':
                print(f"\n📊 Session Statistics:")
                print(f"- Current document: {os.path.basename(pdf_path)}")
                print(f"- Force reprocess: {force_reprocess}")
                print(f"- Index newly created: {chatbot.is_newly_created}")
                continue
            
            if not user_query:
                print("Please enter a valid question or command.")
                continue
            
            print("\n🔄 Processing your question...")
            logger.info(f"Processing query: {user_query}")
            
            # Use LangGraph workflow to process the query
            answer_data = chatbot.ask_question(user_query)
            
            # Display the structured answer
            print("\n📝 Answer:")
            print("-" * 30)
            
            if isinstance(answer_data, dict):
                print(f"Answer: {answer_data.get('answer', 'No answer provided')}")
                
                citations = answer_data.get('citations', [])
                if citations:
                    print(f"\n📖 Sources: {', '.join(citations)}")
                
                confidence = answer_data.get('confidence', 'unknown')
                print(f"🎯 Confidence: {confidence}")
                
                explanation = answer_data.get('explanation', '')
                if explanation and explanation != 'N/A':
                    print(f"💡 Explanation: {explanation}")
            else:
                # Fallback for string responses
                print(answer_data)
                
            print("-" * 30)
    
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot stopped by user.")
        logger.info("Chatbot stopped by user interrupt")
    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(error_msg, exc_info=True)
        print("Please check your configuration and try again.")
        sys.exit(1)
