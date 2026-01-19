"""
Fine-tuning utilities for RAG chatbot with Google Colab integration.
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_core.documents import Document


class FineTuningDataPrep:
    """Prepare training data for fine-tuning language models."""
    
    def __init__(self, output_dir: str = "./fine_tuning_data"):
        """Initialize data preparation utilities.
        
        Args:
            output_dir: Directory to save prepared datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def prepare_qa_data_from_documents(
        self, 
        documents: List[Document], 
        qa_pairs: List[Dict[str, str]],
        format_type: str = "jsonl"
    ) -> str:
        """Prepare question-answer pairs from documents for fine-tuning.
        
        Args:
            documents: List of loaded documents
            qa_pairs: List of question-answer dictionaries
            format_type: Output format ('jsonl', 'csv', 'json')
            
        Returns:
            Path to prepared dataset file
        """
        training_data = []
        
        for qa in qa_pairs:
            # Find relevant documents for this question
            relevant_docs = self._find_relevant_documents(qa["question"], documents)
            
            context = "\n".join([doc.page_content for doc in relevant_docs[:3]])
            
            training_example = {
                "instruction": qa["question"],
                "input": context,
                "output": qa["answer"],
                "context": context
            }
            training_data.append(training_example)
        
        # Save in specified format
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_training_data_{timestamp}.{format_type}"
        filepath = self.output_dir / filename
        
        if format_type == "jsonl":
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format_type == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
        elif format_type == "csv":
            df = pd.DataFrame(training_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"✅ Training data saved to {filepath}")
        print(f"📊 Prepared {len(training_data)} training examples")
        
        return str(filepath)
    
    def prepare_instruction_data(
        self, 
        documents: List[Document],
        instruction_style: str = "alpaca"
    ) -> str:
        """Prepare instruction-following data from documents.
        
        Args:
            documents: List of documents
            instruction_style: Format style ('alpaca', 'chat', 'instruction')
            
        Returns:
            Path to prepared dataset file
        """
        training_data = []
        
        for doc in documents:
            content = doc.page_content
            source = doc.metadata.get("source", "unknown")
            
            # Split content into smaller chunks for instruction generation
            chunks = self._chunk_content(content, chunk_size=500)
            
            for i, chunk in enumerate(chunks):
                if instruction_style == "alpaca":
                    example = {
                        "instruction": f"Based on the following context from {source}, provide a detailed explanation:",
                        "input": chunk,
                        "output": self._generate_explanation(chunk),
                        "text": f"### Instruction:\nBased on the following context from {source}, provide a detailed explanation:\n\n### Input:\n{chunk}\n\n### Response:\n{self._generate_explanation(chunk)}"
                    }
                elif instruction_style == "chat":
                    example = {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant specialized in explaining document content."},
                            {"role": "user", "content": f"Explain this content from {source}: {chunk}"},
                            {"role": "assistant", "content": self._generate_explanation(chunk)}
                        ]
                    }
                else:  # instruction
                    example = {
                        "instruction": f"Explain the following content:",
                        "context": chunk,
                        "response": self._generate_explanation(chunk)
                    }
                
                training_data.append(example)
        
        # Save dataset
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"instruction_data_{timestamp}.jsonl"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Instruction data saved to {filepath}")
        print(f"📊 Prepared {len(training_data)} instruction examples")
        
        return str(filepath)
    
    def create_colab_dataset(
        self, 
        data_file: str,
        colab_notebook_path: str = "./fine_tuning_colab.ipynb"
    ) -> str:
        """Create Google Colab notebook for fine-tuning.
        
        Args:
            data_file: Path to prepared training data
            colab_notebook_path: Path to save Colab notebook
            
        Returns:
            Path to created Colab notebook
        """
        notebook_content = self._generate_colab_notebook(data_file)
        
        with open(colab_notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=2)
        
        print(f"✅ Colab notebook created: {colab_notebook_path}")
        print("📓 Upload this notebook to Google Colab to start fine-tuning")
        
        return colab_notebook_path
    
    def _find_relevant_documents(
        self, 
        question: str, 
        documents: List[Document],
        top_k: int = 3
    ) -> List[Document]:
        """Find documents most relevant to the question."""
        # Simple keyword matching for now
        # In production, you'd use embeddings/similarity search
        question_words = set(question.lower().split())
        
        scored_docs = []
        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(question_words & doc_words)
            scored_docs.append((doc, overlap))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def _chunk_content(self, content: str, chunk_size: int = 500) -> List[str]:
        """Split content into smaller chunks."""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _generate_explanation(self, content: str) -> str:
        """Generate a basic explanation for content (placeholder)."""
        # In production, you'd use a language model to generate better explanations
        sentences = content.split('.')
        if len(sentences) > 1:
            return f"This content discusses {sentences[0].lower()}. Key points include: " + \
                   f". ".join(sentences[1:3]) + "."
        else:
            return f"This content provides information about: {content}"
    
    def _generate_colab_notebook(self, data_file: str) -> Dict[str, Any]:
        """Generate Google Colab notebook content."""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🚀 Fine-Tuning RAG Chatbot Model\n",
                        "\n",
                        "This notebook will guide you through fine-tuning a language model for your RAG chatbot using Google Colab's GPU resources.\n",
                        "\n",
                        "## 📋 Steps:\n",
                        "1. Setup environment\n",
                        "2. Load training data\n",
                        "3. Configure model and training parameters\n",
                        "4. Start fine-tuning\n",
                        "5. Download fine-tuned model"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 🔧 Step 1: Setup Environment"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Install required libraries\n",
                        "!pip install transformers datasets accelerate peft bitsandbytes\n",
                        "!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
                        "\n",
                        "import torch\n",
                        "from transformers import (\n",
                        "    AutoTokenizer, \n",
                        "    AutoModelForCausalLM,\n",
                        "    TrainingArguments,\n",
                        "    Trainer,\n",
                        "    DataCollatorForLanguageModeling\n",
                        ")\n",
                        "from datasets import load_dataset\n",
                        "from peft import LoraConfig, get_peft_model\n",
                        "import json\n",
                        "import os\n",
                        "\n",
                        "print(f\"GPU available: {torch.cuda.is_available()}\")\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 📂 Step 2: Upload and Load Training Data"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Upload your training data file\n",
                        "from google.colab import files\n",
                        "\n",
                        "print(\"Please upload your training data file (JSONL format):\")\n",
                        "uploaded = files.upload()\n",
                        "\n",
                        "# Get the uploaded filename\n",
                        "data_file = list(uploaded.keys())[0]\n",
                        "print(f\"Uploaded file: {data_file}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Load and inspect the training data\n",
                        "def load_training_data(file_path):\n",
                        "    training_data = []\n",
                        "    with open(file_path, 'r', encoding='utf-8') as f:\n",
                        "        for line in f:\n",
                        "            training_data.append(json.loads(line))\n",
                        "    return training_data\n",
                        "\n",
                        "training_data = load_training_data(data_file)\n",
                        "print(f\"Loaded {len(training_data)} training examples\")\n",
                        "\n",
                        "# Display first example\n",
                        "if training_data:\n",
                        "    print(\"\\nFirst training example:\")\n",
                        "    print(json.dumps(training_data[0], indent=2))"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## ⚙️ Step 3: Configure Model and Training"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Model configuration\n",
                        "MODEL_NAME = \"microsoft/DialoGPT-medium\"  # Or \"microsoft/DialoGPT-large\"\n",
                        "MAX_LENGTH = 512\n",
                        "BATCH_SIZE = 4\n",
                        "LEARNING_RATE = 5e-5\n",
                        "NUM_EPOCHS = 3\n",
                        "\n",
                        "# Load tokenizer and model\n",
                        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
                        "model = AutoModelForCausalLM.from_pretrained(\n",
                        "    MODEL_NAME,\n",
                        "    torch_dtype=torch.float16,\n",
                        "    device_map=\"auto\"\n",
                        ")\n",
                        "\n",
                        "# Add padding token if not present\n",
                        "if tokenizer.pad_token is None:\n",
                        "    tokenizer.pad_token = tokenizer.eos_token\n",
                        "\n",
                        "print(f\"Model loaded: {MODEL_NAME}\")\n",
                        "print(f\"Model parameters: {model.num_parameters():,}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Prepare dataset for training\n",
                        "def format_training_example(example):\n",
                        "    \"\"\"Format example for training.\"\"\"\n",
                        "    if \"instruction\" in example:\n",
                        "        # Alpaca format\n",
                        "        text = f\"### Instruction:\\n{example['instruction']}\\n\\n\"\n",
                        "        if \"input\" in example and example[\"input\"]:\n",
                        "            text += f\"### Input:\\n{example['input']}\\n\\n\"\n",
                        "        text += f\"### Response:\\n{example['output']}\"\n",
                        "    else:\n",
                        "        # Simple QA format\n",
                        "        text = f\"Q: {example['question']}\\nA: {example['answer']}\"\n",
                        "    \n",
                        "    return text\n",
                        "\n",
                        "# Tokenize dataset\n",
                        "def tokenize_function(examples):\n",
                        "    texts = [format_training_example(ex) for ex in examples]\n",
                        "    \n",
                        "    tokenized = tokenizer(\n",
                        "        texts,\n",
                        "        truncation=True,\n",
                        "        padding=True,\n",
                        "        max_length=MAX_LENGTH,\n",
                        "        return_tensors=\"pt\"\n",
                        "    )\n",
                        "    \n",
                        "    # For causal LM, we use the input as labels\n",
                        "    tokenized[\"labels\"] = tokenized[\"input_ids\"].clone()\n",
                        "    \n",
                        "    return tokenized\n",
                        "\n",
                        "# Create dataset\n",
                        "from datasets import Dataset\n",
                        "dataset = Dataset.from_list(training_data)\n",
                        "tokenized_dataset = dataset.map(\n",
                        "    lambda x: tokenize_function(x),\n",
                        "    batched=True,\n",
                        "    remove_columns=dataset.column_names\n",
                        ")\n",
                        "\n",
                        "print(f\"Dataset tokenized: {len(tokenized_dataset)} examples\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Configure LoRA for parameter-efficient fine-tuning\n",
                        "lora_config = LoraConfig(\n",
                        "    r=16,  # rank\n",
                        "    lora_alpha=32,\n",
                        "    target_modules=[\"q_proj\", \"v_proj\", \"k_proj\", \"o_proj\"],\n",
                        "    lora_dropout=0.05,\n",
                        "    bias=\"none\",\n",
                        "    task_type=\"CAUSAL_LM\"\n",
                        ")\n",
                        "\n",
                        "# Apply LoRA to model\n",
                        "model = get_peft_model(model, lora_config)\n",
                        "model.print_trainable_parameters()\n",
                        "\n",
                        "# Data collator\n",
                        "data_collator = DataCollatorForLanguageModeling(\n",
                        "    tokenizer=tokenizer,\n",
                        "    mlm=False,\n",
                        ")\n",
                        "\n",
                        "print(\"LoRA configuration applied\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 🏃‍♂️ Step 4: Start Fine-Tuning"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Training arguments\n",
                        "training_args = TrainingArguments(\n",
                        "    output_dir=\"./fine_tuned_model\",\n",
                        "    num_train_epochs=NUM_EPOCHS,\n",
                        "    per_device_train_batch_size=BATCH_SIZE,\n",
                        "    gradient_accumulation_steps=1,\n",
                        "    warmup_steps=100,\n",
                        "    learning_rate=LEARNING_RATE,\n",
                        "    logging_steps=10,\n",
                        "    save_steps=500,\n",
                        "    save_total_limit=2,\n",
                        "    fp16=True,  # Use mixed precision\n",
                        "    dataloader_pin_memory=False,\n",
                        "    remove_unused_columns=False,\n",
                        "    report_to=\"none\",  # Disable wandb/tensorboard\n",
                        ")\n",
                        "\n",
                        "# Create trainer\n",
                        "trainer = Trainer(\n",
                        "    model=model,\n",
                        "    args=training_args,\n",
                        "    train_dataset=tokenized_dataset,\n",
                        "    data_collator=data_collator,\n",
                        ")\n",
                        "\n",
                        "print(\"🚀 Starting fine-tuning...\")\n",
                        "print(f\"Training for {NUM_EPOCHS} epochs\")\n",
                        "print(f\"Batch size: {BATCH_SIZE}\")\n",
                        "print(f\"Learning rate: {LEARNING_RATE}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Start training\n",
                        "trainer.train()\n",
                        "\n",
                        "print(\"✅ Fine-tuning completed!\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 💾 Step 5: Save and Download Model"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Save the fine-tuned model\n",
                        "final_model_path = \"./rag_chatbot_fine_tuned\"\n",
                        "trainer.save_model(final_model_path)\n",
                        "tokenizer.save_pretrained(final_model_path)\n",
                        "\n",
                        "print(f\"✅ Model saved to: {final_model_path}\")\n",
                        "\n",
                        "# Create a zip file for download\n",
                        "import zipfile\n",
                        "import shutil\n",
                        "\n",
                        "zip_filename = \"rag_chatbot_fine_tuned.zip\"\n",
                        "shutil.make_archive(\"rag_chatbot_fine_tuned\", 'zip', final_model_path)\n",
                        "\n",
                        "print(f\"📦 Model zipped as: {zip_filename}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Download the fine-tuned model\n",
                        "from google.colab import files\n",
                        "\n",
                        "print(\"📥 Downloading your fine-tuned model...\")\n",
                        "files.download(zip_filename)\n",
                        "print(\"✅ Download complete! You can now use this model in your RAG chatbot.\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 🎯 Next Steps\n",
                        "\n",
                        "1. **Extract the model**: Unzip the downloaded file in your project\n",
                        "2. **Update your RAG chatbot**: Modify `rag_chat.py` to use the fine-tuned model\n",
                        "3. **Test the model**: Run your chatbot with the new fine-tuned model\n",
                        "\n",
                        "### Example usage in your chatbot:\n",
                        "```python\n",
                        "# In rag_chat.py, update the model initialization:\n",
                        "model_path = \"./rag_chatbot_fine_tuned\"\n",
                        "self.llm = OllamaLLM(model=model_path)  # Or use with transformers directly\n",
                        "```\n",
                        "\n",
                        "## 🎉 Congratulations!\n",
                        "\n",
                        "You've successfully fine-tuned a model for your RAG chatbot! The model should now be better at answering questions based on your specific documents and domain."
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5"
                },
                "colab": {
                    "provenance": [],
                    "gpuType": "T4"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }


class FineTuningManager:
    """Main fine-tuning manager for the RAG chatbot."""
    
    def __init__(self, project_dir: str = "."):
        """Initialize fine-tuning manager.
        
        Args:
            project_dir: Root project directory
        """
        self.project_dir = Path(project_dir)
        self.data_prep = FineTuningDataPrep(
            output_dir=str(self.project_dir / "fine_tuning_data")
        )
        
    def create_fine_tuning_pipeline(
        self, 
        documents: List[Document],
        qa_pairs: Optional[List[Dict[str, str]]] = None,
        create_colab: bool = True
    ) -> Dict[str, str]:
        """Create complete fine-tuning pipeline.
        
        Args:
            documents: List of documents for training
            qa_pairs: Optional QA pairs for supervised fine-tuning
            create_colab: Whether to create Colab notebook
            
        Returns:
            Dictionary with paths to created files
        """
        results = {}
        
        print("🚀 Starting fine-tuning pipeline...")
        
        # Create instruction data
        print("\n📝 Creating instruction data...")
        instruction_file = self.data_prep.prepare_instruction_data(documents)
        results["instruction_data"] = instruction_file
        
        # Create QA data if provided
        if qa_pairs:
            print("\n❓ Creating QA training data...")
            qa_file = self.data_prep.prepare_qa_data_from_documents(
                documents, qa_pairs
            )
            results["qa_data"] = qa_file
        
        # Create Colab notebook
        if create_colab:
            print("\n📓 Creating Google Colab notebook...")
            colab_file = self.data_prep.create_colab_dataset(
                instruction_file,
                colab_notebook_path=str(self.project_dir / "fine_tuning_colab.ipynb")
            )
            results["colab_notebook"] = colab_file
        
        print("\n✅ Fine-tuning pipeline completed!")
        print("📁 Created files:")
        for key, path in results.items():
            print(f"  • {key}: {path}")
        
        return results
    
    def generate_sample_qa_pairs(self, documents: List[Document]) -> List[Dict[str, str]]:
        """Generate sample QA pairs from documents for testing.
        
        Args:
            documents: List of documents
            
        Returns:
            List of QA pairs
        """
        qa_pairs = []
        
        for doc in documents[:5]:  # Generate for first 5 documents
            content = doc.page_content
            source = doc.metadata.get("source", "unknown")
            
            # Simple QA generation (in production, use LLM)
            sentences = content.split('.')
            if len(sentences) > 2:
                question = f"What information is available in {source}?"
                answer = f"The document contains: {sentences[0]}. {sentences[1]}"
                
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "source": source
                })
        
        return qa_pairs