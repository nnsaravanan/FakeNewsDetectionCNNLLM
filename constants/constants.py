MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
KEY = None
with open("../keys/googleapi", 'r') as f:
    KEY = f.read().strip()