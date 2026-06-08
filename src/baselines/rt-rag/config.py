## original settings

# Tree generation parameters for hierarchical QA
TREES_PER_QUESTION = 5           # Number of trees to generate per question (for consensus-based QA)
MAX_TOKENS = 2000                # Maximum number of tokens allowed per tree
DECOMPOSE_TEMPERATURE = 0.8
TOP_P = 1.0 
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
NUM_EXAMPLES = 25                # Number of few-shot examples
MAX_HEIGHT = 4                   # Maximum depth of the generated tree
ENHANCED_RIGHT_SUBTREE = True
RIGHT_SUBTREE_VARIANTS = 1
RIGHT_SUBTREE_TREES_PER_VARIANT = 2
MAX_VARIANTS = 2                  # Original value: 4

# Path to save run-time statistics and logs
STATS_FILE_PATH = ""



# OpenAI-compatible language model API settings
BASE_URL = "https://your-openai-compatible-api-url/v1"
API_KEY = "your-api-key-string"

# Path to save generated dense embeddings
EMBEDDING_DATA = ""

# External reranker service settings (optional)
RANKER_URL = "https://your-ranker-service-url/v1"
RANKER_KEY = "your-ranker-api-key"

# Retrieval configuration
RETRIEVE_TEMPERATURE = 0.3
DATASET = "musique"             # Dataset name (e.g., "musique", "hotpotqa", etc.)
METHOD = "dense"                # Retrieval method: "dense" or "bm25"
CHUNK_SIZE = 200                # Max number of words per chunk
MIN_SENTENCE = 2                # Minimum number of sentences per chunk
OVERLAP = 2                     # Number of overlapping sentences between chunks
TOPK1 = 45                      # Top-K candidates from initial retrieval
TOPK2 = 15                      # Top-K reranked candidates
SAMPLING_ITERATIONS = 5        # Number of sampling iterations for consensus
MAX_ITERATIONS = 4             # Maximum number of iterations for query rewriting

# Root output directory for saving predictions/results
OUTPUT_DIR_ROOT = ""

# Concurrency control
MAX_CONCURRENT = 4              # Maximum number of concurrent QA jobs

# Path to evaluation dataset (in .jsonl format)
DATA_PATH = "/path/to/dataset.jsonl"


