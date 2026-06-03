# ==============================
# EMBEDDING MODEL
# ==============================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


# ==============================
# VECTOR DATABASE
# ==============================

CHROMA_DB_DIR = "data/chroma_db"

COLLECTION_NAME = "multi_pdf_rag"


# ==============================
# RETRIEVAL CONFIGURATION
# ==============================

TOP_K_RESULTS = 5


# ==============================
# FILE STORAGE
# ==============================

UPLOAD_DIR = "data/uploads"


# ==============================
# DOCLING CONFIGURATION
# ==============================

DO_OCR = True

DO_TABLE_STRUCTURE = True

NUM_THREADS = 8