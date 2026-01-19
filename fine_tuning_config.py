# Fine-Tuning Configuration

# Model Configuration
BASE_MODEL = "microsoft/DialoGPT-medium"  # Base model for fine-tuning
FINE_TUNED_MODEL_NAME = "rag_chatbot_fine_tuned"  # Name for fine-tuned model

# Training Parameters
MAX_LENGTH = 512          # Maximum sequence length
BATCH_SIZE = 4            # Training batch size
LEARNING_RATE = 5e-5      # Learning rate
NUM_EPOCHS = 3            # Number of training epochs
WARMUP_STEPS = 100         # Warmup steps
GRADIENT_ACCUMULATION = 1   # Gradient accumulation steps

# LoRA Configuration
LORA_RANK = 16            # LoRA rank
LORA_ALPHA = 32           # LoRA alpha
LORA_DROPOUT = 0.05      # LoRA dropout
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Target modules

# Data Processing
CHUNK_SIZE = 1000         # Document chunk size for training data
CHUNK_OVERLAP = 200      # Chunk overlap
MAX_TRAINING_EXAMPLES = 1000  # Maximum training examples to generate

# Paths Configuration
FINE_TUNING_DATA_DIR = "./fine_tuning_data"
COLAB_NOTEBOOK_PATH = "./fine_tuning_colab.ipynb"
MODEL_OUTPUT_DIR = "./fine_tuned_model"

# Generation Configuration
TEMPERATURE = 0.7          # Generation temperature
TOP_P = 0.9              # Nucleus sampling parameter
TOP_K = 50                # Top-k sampling
MAX_NEW_TOKENS = 256       # Maximum new tokens to generate
DO_SAMPLE = True           # Whether to use sampling

# Hardware Configuration
USE_GPU = True            # Whether to use GPU if available
MIXED_PRECISION = True     # Use mixed precision training
DEVICE_MAP = "auto"        # Device mapping strategy

# Logging and Saving
LOGGING_STEPS = 10        # Steps between logging
SAVE_STEPS = 500          # Steps between model saves
SAVE_TOTAL_LIMIT = 2       # Maximum number of checkpoints to keep

# Data Formats
SUPPORTED_FORMATS = ["jsonl", "json", "csv"]  # Supported export formats
DEFAULT_FORMAT = "jsonl"    # Default export format

# Model Integration
SUPPORTED_PROVIDERS = ["ollama", "huggingface", "transformers"]  # Supported model providers
DEFAULT_PROVIDER = "ollama"  # Default model provider