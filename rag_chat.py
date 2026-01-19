import os
import shutil
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from fine_tuning import FineTuningManager


class ConfigurableDocBot:
    def __init__(self, model_name="cus-qwen", use_fine_tuned=False, fine_tuned_model_path=None):
        """Initialize configurable RAG chatbot
        
        Args:
            model_name: Name of the base model to use
            use_fine_tuned: Whether to use a fine-tuned model
            fine_tuned_model_path: Path to fine-tuned model directory
        """
        self.model_name = model_name
        self.use_fine_tuned = use_fine_tuned
        self.fine_tuned_model_path = fine_tuned_model_path
        self.vectorstore = None
        self.qa_chain = None
        self.current_domain = None
        
        print("🚀 Initializing Configurable Document Bot...")
        print("⏳ Loading embeddings model (first time downloads ~90MB)...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings model loaded!")
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize fine-tuning manager
        self.fine_tuning_manager = FineTuningManager()
        
        print("✅ Bot initialization complete!\n")
    
    def _initialize_llm(self):
        """Initialize language model based on configuration."""
        if self.use_fine_tuned and self.fine_tuned_model_path:
            if os.path.exists(self.fine_tuned_model_path):
                print(f"⏳ Loading fine-tuned model from {self.fine_tuned_model_path}...")
                try:
                    # Try to use with Ollama if fine-tuned model is available there
                    self.llm = OllamaLLM(model=self.fine_tuned_model_path)
                    print(f"✅ Fine-tuned model loaded: {self.fine_tuned_model_path}")
                except Exception as e:
                    print(f"⚠️  Could not load fine-tuned model with Ollama: {e}")
                    print("🔄 Falling back to base model...")
                    self.llm = OllamaLLM(model=self.model_name)
            else:
                print(f"⚠️  Fine-tuned model path not found: {self.fine_tuned_model_path}")
                print("🔄 Using base model instead...")
                self.llm = OllamaLLM(model=self.model_name)
        else:
            self.llm = OllamaLLM(model=self.model_name)
            print(f"✅ Base model loaded: {self.model_name}")
    
    def load_domain(self, domain_name, docs_folder):
        """Load documents for a specific domain/customer"""
        print(f"\n{'='*60}")
        print(f"📂 Loading domain: {domain_name}")
        print(f"📁 Document folder: {docs_folder}")
        print(f"{'='*60}")
        
        self.current_domain = domain_name
        
        # Clean up old vector store
        db_path = f"./chroma_db_{domain_name}"
        if os.path.exists(db_path):
            print(f"🗑️  Removing old vector database...")
            shutil.rmtree(db_path)
        
        # Load documents
        documents = self._load_documents(docs_folder)
        
        if not documents:
            print(f"\n⚠️  WARNING: No documents found in {docs_folder}")
            print(f"ℹ️  The bot will respond that information is not available")
            self.vectorstore = None
            self.qa_chain = None
            return False
        
        # Process documents and create vector database
        self._process_documents(documents, domain_name)
        
        # Setup QA chain
        self._setup_qa_chain()
        
        print(f"\n✅ Domain '{domain_name}' configured successfully!")
        print(f"📊 Ready to answer questions about {len(documents)} documents\n")
        return True
    
    def _load_documents(self, docs_folder):
        """Load all supported document types from folder"""
        documents = []
        folder_path = Path(docs_folder)
        
        if not folder_path.exists():
            print(f"❌ Error: Folder '{docs_folder}' does not exist")
            return documents
        
        print("\n📄 Loading documents...")
        
        # Load markdown and text files
        for ext in ['*.md', '*.txt']:
            for file_path in folder_path.rglob(ext):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():  # Only if not empty
                            doc = Document(
                                page_content=content,
                                metadata={"source": str(file_path)}
                            )
                            documents.append(doc)
                            print(f"  ✓ Loaded {file_path.name}")
                except Exception as e:
                    print(f"  ⚠️  Error loading {file_path.name}: {e}")
        
        # Load PDF files
        try:
            from pypdf import PdfReader
            for file_path in folder_path.rglob('*.pdf'):
                try:
                    reader = PdfReader(str(file_path))
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    if text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={"source": str(file_path)}
                        )
                        documents.append(doc)
                        print(f"  ✓ Loaded {file_path.name}")
                except Exception as e:
                    print(f"  ⚠️  Error loading {file_path.name}: {e}")
        except ImportError:
            print("  ℹ️  pypdf not available, skipping PDF files")
        
        if documents:
            print(f"\n📚 Total documents loaded: {len(documents)}")
        
        return documents
    
    def _process_documents(self, documents, domain_name):
        """Split documents into chunks and create vector database"""
        print("\n⚙️  Processing documents...")
        
        # Split documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"  ✓ Split into {len(texts)} chunks")
        
        # Create vector database
        print(f"  ⏳ Creating vector database...")
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=f"./chroma_db_{domain_name}"
        )
        print(f"  ✓ Vector database created and persisted")
    
    def _setup_qa_chain(self):
        """Setup the question-answering chain"""
        if self.vectorstore is None:
            print("⚠️  No vector store - QA chain not created")
            return
        
        print("  ⏳ Setting up QA chain...")
        
        # Create retriever (finds relevant chunks)
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )
        
        # Create prompt template
        prompt_template = """CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY the information in the CONTEXT above
2. Follow a step-by-step format from your system prompt
3. If the answer is not in the CONTEXT, respond: "This information is not available to me. If you want, I can contact a human agent!"
4. Cite specific sections or sources when possible

YOUR ANSWER:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("  ✓ QA chain ready")
    
    def ask(self, question):
        """Ask a question about the loaded documents"""
        if not self.current_domain:
            return "❌ No domain loaded. Use load_domain() first."
        
        if self.qa_chain is None:
            return "This information is not available to me. If you want, I can contact a human agent!"
        
        print(f"\n{'─'*60}")
        print(f"🔍 Question: {question}")
        print(f"📂 Domain: {self.current_domain}")
        print("⏳ Searching documents...")
        
        try:
            # Query the chain
            result = self.qa_chain.invoke({"query": question})
            
            answer = result["result"]
            
            # Display source documents
            if result.get("source_documents"):
                print(f"\n📄 Sources used:")
                for i, doc in enumerate(result["source_documents"][:3], 1):
                    source = doc.metadata.get("source", "Unknown")
                    snippet = doc.page_content[:150].replace("\n", " ")
                    print(f"  {i}. {os.path.basename(source)}")
                    print(f"     Preview: {snippet}...")
            
            print(f"{'─'*60}\n")
            return answer
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            return f"Error processing question: {str(e)}"
    
    def switch_domain(self, domain_name, docs_folder):
        """Switch to a different domain/customer"""
        print(f"\n🔄 Switching from '{self.current_domain}' to '{domain_name}'...")
        return self.load_domain(domain_name, docs_folder)
    
    def prepare_fine_tuning_data(self, qa_pairs=None):
        """Prepare data for fine-tuning using current documents.
        
        Args:
            qa_pairs: Optional list of QA pairs for supervised fine-tuning
            
        Returns:
            Dictionary with paths to created files
        """
        if not self.current_domain or not self.vectorstore:
            print("❌ No domain loaded. Please load a domain first.")
            return {}
        
        print(f"\n🔧 Preparing fine-tuning data for domain: {self.current_domain}")
        
        # Get documents from current vector store
        documents = self._get_current_documents()
        
        if not documents:
            print("❌ No documents found for fine-tuning.")
            return {}
        
        # Generate sample QA pairs if none provided
        if not qa_pairs:
            print("📝 Generating sample QA pairs...")
            qa_pairs = self.fine_tuning_manager.generate_sample_qa_pairs(documents)
        
        # Create fine-tuning pipeline
        results = self.fine_tuning_manager.create_fine_tuning_pipeline(
            documents=documents,
            qa_pairs=qa_pairs,
            create_colab=True
        )
        
        return results
    
    def _get_current_documents(self):
        """Get documents from current vector store."""
        if not self.vectorstore:
            return []
        
        try:
            # Get all documents from vector store
            collection = self.vectorstore._collection
            results = collection.get(include=['documents', 'metadatas'])
            
            documents = []
            if results['documents']:
                for i, doc_content in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}
                    doc = Document(page_content=doc_content, metadata=metadata)
                    documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []
    
    def switch_to_fine_tuned_model(self, model_path):
        """Switch to using a fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model
        """
        if not os.path.exists(model_path):
            print(f"❌ Fine-tuned model not found: {model_path}")
            return False
        
        print(f"🔄 Switching to fine-tuned model: {model_path}")
        
        old_model = self.model_name
        old_use_fine_tuned = self.use_fine_tuned
        old_fine_tuned_path = self.fine_tuned_model_path
        
        self.use_fine_tuned = True
        self.fine_tuned_model_path = model_path
        
        try:
            self._initialize_llm()
            # Re-setup QA chain with new model
            if self.vectorstore:
                self._setup_qa_chain()
            
            print("✅ Successfully switched to fine-tuned model!")
            return True
        except Exception as e:
            print(f"❌ Error switching to fine-tuned model: {e}")
            # Revert to old model
            self.model_name = old_model
            self.use_fine_tuned = old_use_fine_tuned
            self.fine_tuned_model_path = old_fine_tuned_path
            self._initialize_llm()
            return False


def main():
    """Interactive CLI for chatting with documents"""
    print("\n" + "="*60)
    print("  CONFIGURABLE DOCUMENT CHATBOT")
    print("="*60 + "\n")
    
    # Ask about fine-tuning
    use_fine_tuned = input("Use fine-tuned model? (y/N): ").strip().lower() == 'y'
    fine_tuned_path = None
    
    if use_fine_tuned:
        fine_tuned_path = input("Enter path to fine-tuned model: ").strip()
        if not fine_tuned_path:
            use_fine_tuned = False
    
    bot = ConfigurableDocBot(
        model_name="cus-qwen",
        use_fine_tuned=use_fine_tuned,
        fine_tuned_model_path=fine_tuned_path
    )
    
    # Ask user which domain to load
    print("Available domain folders:")
    print("  - Documents/Medical")
    print("  - Documents/Legal")
    print("  - Documents/Tech")
    print()
    
    domain_name = input("Enter domain name (e.g., 'medical'): ").strip()
    if not domain_name:
        domain_name = "medical"
    
    docs_folder = input(f"Enter docs folder (default: ./Documents/{domain_name}): ").strip()
    if not docs_folder:
        docs_folder = f"./Documents/{domain_name}"
    
    # Load the domain
    success = bot.load_domain(domain_name, docs_folder)
    
    if not success:
        print("\n⚠️  Failed to load domain or no documents found.")
        print("Continuing anyway - bot will say info is not available.")
    
    # Chat loop
    print(f"\n💬 Chat with '{domain_name}' documents")
    print("Commands:")
    print("  • Type your question to ask")
    print("  • 'switch' - change to different domain")
    print("  • 'fine-tune' - prepare fine-tuning data")
    print("  • 'use-fine-tuned' - switch to fine-tuned model")
    print("  • 'quit' - exit")
    print()
    
    while True:
        try:
            user_input = input(f"[{domain_name}] You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if user_input.lower() == 'switch':
                new_domain = input("New domain name: ").strip()
                new_folder = input(f"Docs folder (default: ./Documents/{new_domain}): ").strip()
                if not new_folder:
                    new_folder = f"./Documents/{new_domain}"
                bot.switch_domain(new_domain, new_folder)
                domain_name = new_domain
                continue
            
            if user_input.lower() == 'fine-tune':
                qa_pairs_input = input("Enter path to QA pairs file (optional, press Enter to auto-generate): ").strip()
                qa_pairs = None
                if qa_pairs_input and os.path.exists(qa_pairs_input):
                    # Load QA pairs from file if provided
                    import json
                    with open(qa_pairs_input, 'r') as f:
                        qa_pairs = json.load(f)
                
                results = bot.prepare_fine_tuning_data(qa_pairs)
                print("\n📁 Fine-tuning files created:")
                for key, path in results.items():
                    print(f"  • {key}: {path}")
                print("\n🎯 Next steps:")
                print("  1. Upload the Colab notebook to Google Colab")
                print("  2. Run the fine-tuning process")
                print("  3. Download the fine-tuned model")
                print("  4. Use 'use-fine-tuned' command to switch to it")
                continue
            
            if user_input.lower() == 'use-fine-tuned':
                model_path = input("Enter path to fine-tuned model: ").strip()
                if model_path:
                    success = bot.switch_to_fine_tuned_model(model_path)
                    if success:
                        print("🎉 Successfully switched to fine-tuned model!")
                    else:
                        print("❌ Failed to switch to fine-tuned model.")
                continue
            
            if not user_input:
                continue
            
            # Ask the question
            answer = bot.ask(user_input)
            print(f"\n🤖 Assistant:\n{answer}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()