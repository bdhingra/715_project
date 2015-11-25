# Set global params here

# Number of epochs
NUM_EPOCHS = 20
# Batch size
N_BATCH = 128
# Max sequence length
MAX_LENGTH = 145
# Number of unique characters
N_CHAR = 1000
# Dimensionality of character lookup
CHAR_DIM = 100
# Minimum levenshtein distance between t_pos hashtags and t_neg hashtags
MIN_LEV_DIST = 5
# Maximum number of triples generated for each hashtag
MAX_TRIPLES_PER_HASHTAG = 1000
# Initialization scale
SCALE = 0.1
# Dimensionality of C2W hidden states
C2W_HDIM = 150
# Dimensionality of word vectors
WDIM = 100
# Gap parameter
M = 1
# Learning rate
LEARNING_RATE = .0005
# Display frequency
DISPF = 10
# Save frequency
SAVEF = 1000
# Validation set
N_VAL = 5000000
# Regularization
REGULARIZATION = 1e-4
# Debug mode
DEBUG = False
# Reload model
RELOAD = True
