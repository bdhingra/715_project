# Set global params here

# Number of epochs
NUM_EPOCHS = 50
# Batch size
N_BATCH = 64
# Max sequence length
MAX_LENGTH = 145
# Max number of characters in word
MAX_WORD_LENGTH = 8
# Max number of words in a tweet
MAX_SEQ_LENGTH = 20
# Number of unique characters
N_CHAR = 1000
# Dimensionality of character lookup
CHAR_DIM = 150
# Minimum levenshtein distance between t_pos hashtags and t_neg hashtags
MIN_LEV_DIST = 5
# Maximum number of triples generated for each hashtag
MAX_TRIPLES_PER_HASHTAG = 1000
# Initialization scale
SCALE = 0.1
# Dimensionality of C2W hidden states
C2W_HDIM = 100
# Dimensionality of word vectors
WDIM = 100
# Dimensionality of W2S hidden states
W2S_HDIM = 100
# Dimensionality of sequence
SDIM = 100
# Gap parameter
M = 0.5
# Learning rate
LEARNING_RATE = 0.1
# Display frequency
DISPF = 5
# Save frequency
SAVEF = 1000
# Validation set
#N_VAL = 5000000
# Regularization
REGULARIZATION = 1e-5
# Debug mode
DEBUG = False
# Reload model
RELOAD_MODEL = False
# Reload data
RELOAD_DATA = False
# Number of attempts to find negative tweet sample
ATTEMPTS = 15
# Momentum
MOMENTUM = 0.99
# learning schedule
USE_SCHEDULE = True
# gradient clipping
GRAD_CLIP = 10
