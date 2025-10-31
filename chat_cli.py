#!/usr/bin/env python3
"""
Simple CLI interface for the PDF Chatbot.
This script provides an easy-to-use command line interface for users.
"""

import os
import sys
import argparse
from pathlib import Path
from advanced_pdf_bot import PDFChatbot, setup_logging


def find_pdf_files(directory: str) -> list:
    """Find all PDF files in a directory."""
    pdf_files = []
    for file_path in Path(directory).glob("*.pdf"):
        pdf_files.append(str(file_path))
    return pdf_files


def select_pdf_interactively() -> str:
    """Allow user to select a PDF file interactively."""
    print("🔍 Looking for PDF files...")
    
    # Check current directory first
    current_dir_pdfs = find_pdf_files(".")
    
    if current_dir_pdfs:
        print(f"\n📁 Found {len(current_dir_pdfs)} PDF files in current directory:")
        for i, pdf_file in enumerate(current_dir_pdfs, 1):
            file_size = os.path.getsize(pdf_file) / (1024 * 1024)  # Size in MB
            print(f"  {i}. {os.path.basename(pdf_file)} ({file_size:.1f}MB)")
        
        print(f"  {len(current_dir_pdfs) + 1}. Browse for another file")
        print("  0. Exit")
        
        while True:
            try:
                choice = input(f"\nSelect a file (1-{len(current_dir_pdfs) + 1}, 0 to exit): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    sys.exit(0)
                elif choice == str(len(current_dir_pdfs) + 1):
                    break  # Go to manual file selection
                else:
                    file_index = int(choice) - 1
                    if 0 <= file_index < len(current_dir_pdfs):
                        return current_dir_pdfs[file_index]
                    else:
                        print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    # Manual file path entry
    print("\n📂 Enter the path to your PDF file:")
    while True:
        pdf_path = input("PDF file path: ").strip()
        
        if not pdf_path:
            print("❌ Please enter a valid file path.")
            continue
        
        # Remove quotes if user added them
        pdf_path = pdf_path.strip('"').strip("'")
        
        # Expand user path (e.g., ~/Documents becomes /home/user/Documents)
        pdf_path = os.path.expanduser(pdf_path)
        
        if os.path.exists(pdf_path):
            if pdf_path.lower().endswith('.pdf'):
                return pdf_path
            else:
                print("❌ File must be a PDF (.pdf extension required).")
                continue
        else:
            print(f"❌ File not found: {pdf_path}")
            continue


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="PDF Chatbot - Ask questions about your PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat_cli.py                           # Interactive mode
  python chat_cli.py --pdf document.pdf       # Process specific file
  python chat_cli.py --pdf document.pdf --force-reprocess  # Force reprocessing
        """
    )
    
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
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🤖 PDF Chatbot - Interactive CLI")
    print("=" * 40)
    
    # Get PDF file path
    pdf_path = args.pdf
    
    if not pdf_path:
        pdf_path = select_pdf_interactively()
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        sys.exit(1)
    
    print(f"\n📄 Processing: {os.path.basename(pdf_path)}")
    print(f"📁 Full path: {pdf_path}")
    
    try:
        # Initialize and setup chatbot
        print("🔄 Initializing chatbot...")
        chatbot = PDFChatbot(pdf_path, force_reprocess=args.force_reprocess)
        chatbot.setup_index()
        
        print("📊 Processing document...")
        chatbot.process_document()
        
        print("✅ Chatbot ready!")
        
        # Show usage instructions
        print("\n💬 How to use:")
        print("• Type any question about the document")
        print("• Special commands:")
        print("  - 'help' or 'h' - Show help")
        print("  - 'info' - Document information")
        print("  - 'quit', 'exit', 'q' - Exit chatbot")
        print("-" * 40)
        
        # Interactive loop
        while True:
            try:
                user_input = input("\n❓ Your question: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using PDF Chatbot!")
                    break
                
                elif user_input.lower() in ['help', 'h']:
                    print("\n📖 Help:")
                    print("• Ask questions about the content of your PDF")
                    print("• The chatbot will provide answers with page references")
                    print("• Type 'info' to see document details")
                    print("• Type 'quit' to exit")
                    continue
                
                elif user_input.lower() == 'info':
                    print(f"\n📄 Document Information:")
                    print(f"• File: {os.path.basename(pdf_path)}")
                    print(f"• Size: {os.path.getsize(pdf_path) / (1024*1024):.1f}MB")
                    print(f"• Document ID: {chatbot.doc_id}")
                    if chatbot.document_summary:
                        print(f"• Summary: {chatbot.document_summary[:150]}...")
                    continue
                
                # Process the question
                print("🔄 Processing...")
                answer_data = chatbot.ask_question(user_input)
                
                # Display answer
                print("\n📝 Answer:")
                print("-" * 30)
                
                if isinstance(answer_data, dict):
                    print(answer_data.get('answer', 'No answer available'))
                    
                    citations = answer_data.get('citations', [])
                    if citations:
                        print(f"\n📖 Sources: {', '.join(citations)}")
                    
                    confidence = answer_data.get('confidence', 'unknown')
                    print(f"🎯 Confidence: {confidence}")
                    
                else:
                    print(answer_data)
                
                print("-" * 30)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chatbot stopped.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    except Exception as e:
        print(f"\n❌ Failed to initialize chatbot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
