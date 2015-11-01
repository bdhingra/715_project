'''
Skip-thoughts encoder decoder implemented with C2W embeddings for words
'''

import numpy as np
import lasagne
import theano
import theano.tensor as T
import random

# Number of epochs
NUM_EPOCHS = 1
# Batch size
N_BATCH = 1
# Max sequence length
MAX_LENGTH = 4
# Number of unique characters
N_CHAR = 5
# Dimensionality of character lookup
CHAR_DIM = 1
# Initialization scale
SCALE = 0.1
# Dimensionality of C2W hidden states
C2W_HDIM = 1
# Dimensionality of word vectors
WDIM = 1

def main(num_epochs=NUM_EPOCHS):
    print("Building network...")

    # Input layer over source characters
    l_in_source = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_LENGTH,1))
    # Input layer over before-context characters
    l_in_before = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_LENGTH,1))
    # Input layer over after-context characters
    l_in_after = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_LENGTH,1))

    # Shared variables for C2W
    np.random.seed(0)
    Wc = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(N_CHAR,CHAR_DIM)).astype('float32'))

    W_c2w_f_r = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'))
    W_c2w_f_z = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'))
    W_c2w_f_h = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'))
    b_c2w_f_r = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'))
    b_c2w_f_z = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'))
    b_c2w_f_h = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'))
    U_c2w_f_r = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'))
    U_c2w_f_z = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'))
    U_c2w_f_h = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'))

    W_c2w_b_r = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'))
    W_c2w_b_z = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'))
    W_c2w_b_h = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'))
    b_c2w_b_r = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'))
    b_c2w_b_z = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'))
    b_c2w_b_h = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'))
    U_c2w_b_r = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'))
    U_c2w_b_z = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'))
    U_c2w_b_h = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'))

    W_c2w_df = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'))
    W_c2w_db = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'))
    b_c2w_df = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(WDIM)).astype('float32'))
    b_c2w_db = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(WDIM)).astype('float32'))

    # C2W
    # lookup for source
    l_clookup_source = lasagne.layers.EmbeddingLayer(l_in_source, input_size=N_CHAR, output_size=CHAR_DIM, W=Wc)
    # lookup for before
    l_clookup_before = lasagne.layers.EmbeddingLayer(l_in_before, input_size=N_CHAR, output_size=CHAR_DIM, W=Wc)
    # lookup for after
    l_clookup_after = lasagne.layers.EmbeddingLayer(l_in_after, input_size=N_CHAR, output_size=CHAR_DIM, W=Wc)

    # f-GRU
    c2w_f_reset = lasagne.layers.Gate(W_in=W_c2w_f_r, W_hid=U_c2w_f_r, W_cell=None, b=b_c2w_f_r, nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_update = lasagne.layers.Gate(W_in=W_c2w_f_z, W_hid=U_c2w_f_z, W_cell=None, b=b_c2w_f_z, nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_hidden = lasagne.layers.Gate(W_in=W_c2w_f_h, W_hid=U_c2w_f_h, W_cell=None, b=b_c2w_f_h, nonlinearity=lasagne.nonlinearities.sigmoid)
    # f-GRU for source
    l_fgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_f_reset, updategate=c2w_f_update, hidden_update=c2w_f_hidden, hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=None)
    # f-GRU for before
    l_fgru_before = lasagne.layers.GRULayer(l_clookup_before, C2W_HDIM, resetgate=c2w_f_reset, updategate=c2w_f_update, hidden_update=c2w_f_hidden, hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=None)
    # f-GRU for after
    l_fgru_after = lasagne.layers.GRULayer(l_clookup_after, C2W_HDIM, resetgate=c2w_f_reset, updategate=c2w_f_update, hidden_update=c2w_f_hidden, hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=None)

    # b-GRU
    c2w_b_reset = lasagne.layers.Gate(W_in=W_c2w_b_r, W_hid=U_c2w_b_r, W_cell=None, b=b_c2w_b_r, nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_update = lasagne.layers.Gate(W_in=W_c2w_b_z, W_hid=U_c2w_b_z, W_cell=None, b=b_c2w_b_z, nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_hidden = lasagne.layers.Gate(W_in=W_c2w_b_h, W_hid=U_c2w_b_h, W_cell=None, b=b_c2w_b_h, nonlinearity=lasagne.nonlinearities.sigmoid)
    # b-GRU for source
    l_bgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_b_reset, updategate=c2w_b_update, hidden_update=c2w_b_hidden, hid_init=lasagne.init.Constant(0.), backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=None)
    # b-GRU for before
    l_bgru_before = lasagne.layers.GRULayer(l_clookup_before, C2W_HDIM, resetgate=c2w_b_reset, updategate=c2w_b_update, hidden_update=c2w_b_hidden, hid_init=lasagne.init.Constant(0.), backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=None)
    # b-GRU for after
    l_bgru_after = lasagne.layers.GRULayer(l_clookup_after, C2W_HDIM, resetgate=c2w_b_reset, updategate=c2w_b_update, hidden_update=c2w_b_hidden, hid_init=lasagne.init.Constant(0.), backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=None)

    # Slice final states
    l_f_source = lasagne.layers.SliceLayer(l_fgru_source, -1, 1)
    l_f_before = lasagne.layers.SliceLayer(l_fgru_before, -1, 1)
    l_f_after = lasagne.layers.SliceLayer(l_fgru_after, -1, 1)
    l_b_source = lasagne.layers.SliceLayer(l_bgru_source, 0, 1)
    l_b_before = lasagne.layers.SliceLayer(l_bgru_before, 0, 1)
    l_b_after = lasagne.layers.SliceLayer(l_bgru_after, 0, 1)

    # Dense layer
    l_fdense_source = lasagne.layers.DenseLayer(l_f_source, WDIM, W=W_c2w_df, b=b_c2w_df, nonlinearity=None)
    l_fdense_before = lasagne.layers.DenseLayer(l_f_before, WDIM, W=W_c2w_df, b=b_c2w_df, nonlinearity=None)
    l_fdense_after = lasagne.layers.DenseLayer(l_f_after, WDIM, W=W_c2w_df, b=b_c2w_df, nonlinearity=None)
    l_bdense_source = lasagne.layers.DenseLayer(l_b_source, WDIM, W=W_c2w_db, b=b_c2w_db, nonlinearity=None)
    l_bdense_before = lasagne.layers.DenseLayer(l_b_before, WDIM, W=W_c2w_db, b=b_c2w_db, nonlinearity=None)
    l_bdense_after = lasagne.layers.DenseLayer(l_b_after, WDIM, W=W_c2w_db, b=b_c2w_db, nonlinearity=None)
    l_c2w_source = lasagne.layers.ElemwiseSumLayer([l_fdense_source, l_bdense_source], coeffs=1)
    l_c2w_before = lasagne.layers.ElemwiseSumLayer([l_fdense_before, l_bdense_before], coeffs=1)
    l_c2w_after = lasagne.layers.ElemwiseSumLayer([l_fdense_after, l_bdense_after], coeffs=1)

    # Word embeddings
    source = T.itensor3()
    before = T.itensor3()
    after = T.itensor3()
    source_emb = lasagne.layers.get_output(l_c2w_source, source)
    before_emb = lasagne.layers.get_output(l_c2w_before, before)
    after_emb = lasagne.layers.get_output(l_c2w_after, after)

    # Theano function
    c2w_source = theano.function([source], source_emb)
    c2w_before = theano.function([before], before_emb)
    c2w_after = theano.function([after], after_emb)

    # Test
    s = np.ones((1,4,1),dtype=np.int32)
    b = np.ones((1,4,1),dtype=np.int32)
    a = np.ones((1,4,1),dtype=np.int32)
    print "Input - \n"
    print str(s)
    print str(b)
    print str(a)
    emb_source = c2w_source(s)
    emb_before = c2w_before(b)
    emb_after = c2w_after(a)
    print "Output - \n"
    print str(emb_source)
    print str(emb_before)
    print str(emb_after)

if __name__ == '__main__':
    main()
