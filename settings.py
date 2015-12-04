# Set global params here

# Number of epochs
NUM_EPOCHS = 10
# Batch size
N_BATCH = 128
# Max sequence length
MAX_LENGTH = 145
# Max number of characters in word
MAX_WORD_LENGTH = 4
# Max number of words in a tweet
MAX_SEQ_LENGTH = 5
# Number of unique characters
N_CHAR = 1000
# Dimensionality of character lookup
CHAR_DIM = 3
# Minimum levenshtein distance between t_pos hashtags and t_neg hashtags
MIN_LEV_DIST = 5
# Maximum number of triples generated for each hashtag
MAX_TRIPLES_PER_HASHTAG = 1000
# Initialization scale
SCALE = 0.1
# Dimensionality of C2W hidden states
C2W_HDIM = 3
# Dimensionality of word vectors
WDIM = 3
# Dimensionality of W2S hidden states
W2S_HDIM = 3
# Dimensionality of sequence
SDIM = 3
# Gap parameter
M = 0.5
# Learning rate
LEARNING_RATE = 0.001
# Display frequency
DISPF = 1
# Save frequency
SAVEF = 1000
# Validation set
N_VAL = 5000000
# Regularization
REGULARIZATION = 0
# Debug mode
DEBUG = True
# Reload model
RELOAD_MODEL = False
# Reload data
RELOAD_DATA = True
# Number of attempts to find negative tweet sample
ATTEMPTS = 15
