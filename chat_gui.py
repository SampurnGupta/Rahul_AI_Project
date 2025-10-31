#!/usr/bin/env python3
"""
Simple GUI interface for the PDF Chatbot using tkinter.
This script provides an easy-to-use graphical interface for users.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
from advanced_pdf_bot import PDFChatbot, setup_logging


class PDFChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Chatbot - Interactive GUI")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Variables
        self.chatbot = None
        self.pdf_path = tk.StringVar()
        self.processing = False
        
        self.setup_ui()
        setup_logging("INFO")
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 PDF Chatbot", font=("TkDefaultFont", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="📄 Select PDF Document", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="PDF File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.file_entry = ttk.Entry(file_frame, textvariable=self.pdf_path, width=50)
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.browse_button = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        self.browse_button.grid(row=0, column=2)
        
        self.load_button = ttk.Button(file_frame, text="Load PDF", command=self.load_pdf, style="Accent.TButton")
        self.load_button.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        # Status section
        self.status_var = tk.StringVar(value="Select a PDF file to get started")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=2, column=0, columnspan=3, sticky=tk.W)
        
        # Chat section
        chat_frame = ttk.LabelFrame(main_frame, text="💬 Chat with your PDF", padding="10")
        chat_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            width=70, 
            height=15,
            state=tk.DISABLED,
            font=("TkDefaultFont", 10)
        )
        self.chat_display.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Question input
        question_frame = ttk.Frame(chat_frame)
        question_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        question_frame.columnconfigure(0, weight=1)
        
        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(question_frame, textvariable=self.question_var, font=("TkDefaultFont", 10))
        self.question_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.question_entry.bind("<Return>", lambda e: self.ask_question())
        
        self.ask_button = ttk.Button(question_frame, text="Ask", command=self.ask_question, style="Accent.TButton")
        self.ask_button.grid(row=0, column=1)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initially disable chat elements
        self.toggle_chat_elements(False)
        
        # Add some example text
        self.add_to_chat("🤖 Welcome to PDF Chatbot!", "system")
        self.add_to_chat("Select a PDF file and click 'Load PDF' to get started.", "system")
    
    def browse_file(self):
        """Open file dialog to select PDF."""
        filename = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        
        if filename:
            self.pdf_path.set(filename)
    
    def toggle_chat_elements(self, enabled):
        """Enable or disable chat interface elements."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.question_entry.config(state=state)
        self.ask_button.config(state=state)
    
    def add_to_chat(self, message, sender="user"):
        """Add message to chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        if sender == "user":
            prefix = "❓ You: "
            self.chat_display.insert(tk.END, f"{prefix}{message}\n\n")
        elif sender == "bot":
            prefix = "🤖 Bot: "
            self.chat_display.insert(tk.END, f"{prefix}{message}\n\n")
        else:  # system
            self.chat_display.insert(tk.END, f"{message}\n\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def update_status(self, message, color="blue"):
        """Update status message."""
        self.status_var.set(message)
        self.status_label.config(foreground=color)
    
    def load_pdf(self):
        """Load and process the selected PDF."""
        pdf_path = self.pdf_path.get().strip()
        
        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF file first.")
            return
        
        if not os.path.exists(pdf_path):
            messagebox.showerror("Error", f"File not found: {pdf_path}")
            return
        
        if not pdf_path.lower().endswith('.pdf'):
            messagebox.showerror("Error", "Please select a PDF file (.pdf extension required).")
            return
        
        # Start processing in a separate thread
        self.processing = True
        self.load_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.progress.start()
        
        threading.Thread(target=self._process_pdf, args=(pdf_path,), daemon=True).start()
    
    def _process_pdf(self, pdf_path):
        """Process PDF in background thread."""
        try:
            self.root.after(0, lambda: self.update_status("Initializing chatbot...", "orange"))
            self.root.after(0, lambda: self.add_to_chat(f"🔄 Loading PDF: {os.path.basename(pdf_path)}", "system"))
            
            # Create chatbot
            self.chatbot = PDFChatbot(pdf_path, force_reprocess=False)
            
            self.root.after(0, lambda: self.update_status("Setting up index...", "orange"))
            self.chatbot.setup_index()
            
            self.root.after(0, lambda: self.update_status("Processing document...", "orange"))
            self.chatbot.process_document()
            
            # Success
            self.root.after(0, self._on_pdf_loaded)
            
        except Exception as e:
            error_msg = f"Error loading PDF: {str(e)}"
            self.root.after(0, lambda: self._on_pdf_error(error_msg))
    
    def _on_pdf_loaded(self):
        """Called when PDF is successfully loaded."""
        self.processing = False
        self.progress.stop()
        self.load_button.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        self.toggle_chat_elements(True)
        
        self.update_status("✅ PDF loaded successfully! You can now ask questions.", "green")
        self.add_to_chat("✅ PDF processed successfully! Ask me any questions about the document.", "system")
        
        # Focus on question entry
        self.question_entry.focus()
    
    def _on_pdf_error(self, error_msg):
        """Called when PDF loading fails."""
        self.processing = False
        self.progress.stop()
        self.load_button.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        
        self.update_status("❌ Error loading PDF", "red")
        self.add_to_chat(f"❌ {error_msg}", "system")
        messagebox.showerror("Error", error_msg)
    
    def ask_question(self):
        """Ask a question to the chatbot."""
        if not self.chatbot:
            messagebox.showwarning("Warning", "Please load a PDF file first.")
            return
        
        question = self.question_var.get().strip()
        if not question:
            return
        
        # Clear question entry
        self.question_var.set("")
        
        # Add question to chat
        self.add_to_chat(question, "user")
        
        # Disable input while processing
        self.ask_button.config(state=tk.DISABLED)
        self.question_entry.config(state=tk.DISABLED)
        self.update_status("🔄 Processing your question...", "orange")
        
        # Process question in background thread
        threading.Thread(target=self._process_question, args=(question,), daemon=True).start()
    
    def _process_question(self, question):
        """Process question in background thread."""
        try:
            answer_data = self.chatbot.ask_question(question)
            self.root.after(0, lambda: self._on_answer_received(answer_data))
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.root.after(0, lambda: self._on_question_error(error_msg))
    
    def _on_answer_received(self, answer_data):
        """Called when answer is received."""
        self.ask_button.config(state=tk.NORMAL)
        self.question_entry.config(state=tk.NORMAL)
        self.update_status("✅ Ready for your next question", "green")
        
        # Format and display answer
        if isinstance(answer_data, dict):
            answer = answer_data.get('answer', 'No answer available')
            citations = answer_data.get('citations', [])
            confidence = answer_data.get('confidence', 'unknown')
            
            response = answer
            if citations:
                response += f"\n\n📖 Sources: {', '.join(citations)}"
            response += f"\n🎯 Confidence: {confidence}"
            
            self.add_to_chat(response, "bot")
        else:
            self.add_to_chat(str(answer_data), "bot")
        
        # Focus back on question entry
        self.question_entry.focus()
    
    def _on_question_error(self, error_msg):
        """Called when question processing fails."""
        self.ask_button.config(state=tk.NORMAL)
        self.question_entry.config(state=tk.NORMAL)
        self.update_status("❌ Error processing question", "red")
        
        self.add_to_chat(f"❌ {error_msg}", "system")
        self.question_entry.focus()


def main():
    """Main function to run the GUI."""
    try:
        root = tk.Tk()
        app = PDFChatbotGUI(root)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
