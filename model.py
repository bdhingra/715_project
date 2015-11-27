import numpy as np
import lasagne
import theano
import theano.tensor as T
import random
import sys

from collections import OrderedDict
from settings import N_BATCH, MAX_LENGTH, N_CHAR, CHAR_DIM, SCALE, C2W_HDIM, WDIM, MAX_WORD_LENGTH, MAX_SEQ_LENGTH, W2S_HDIM, SDIM

def char2word2vec(batch, mask, params, n_char=N_CHAR):
    '''
    Hierarchical GRU
    '''
    # Reshape to make a batch of words
    words = batch.reshape((N_BATCH*MAX_SEQ_LENGTH,MAX_WORD_LENGTH,1),ndim=3)
    words_mask = mask.reshape((N_BATCH*MAX_SEQ_LENGTH,MAX_WORD_LENGTH),ndim=3)

    # word embeddings
    word_emb, c2w_net = char2word(words,words_mask,params['c2w'],n_char)

    # Reshape back to batch of sequences
    seqs = word_emb.reshape((N_BATCH,MAX_SEQ_LENGTH,WDIM),ndim=3)
    seqs_mask = mask[:,:,1]

    # sequence embeddings
    seq_emb, w2s_net = word2seq(seqs,seqs_mask,params['w2s'])

    return seq_emb, c2w_net, w2s_net

def word2seq(seqs,mask,params):
    '''
    Sequence embeddings by composing word embeddings
    '''
    # Input layer over characters
    l_in_source = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_SEQ_LENGTH,WDIM), input_var=seqs, name='input')

    # Mask layer for variable length sequences
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_SEQ_LENGTH), input_var=mask, name='mask')

    # f-GRU
    c2w_f_reset = lasagne.layers.Gate(W_in=params['W_f_r'], W_hid=params['U_f_r'], W_cell=None, b=params['b_f_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_update = lasagne.layers.Gate(W_in=params['W_f_z'], W_hid=params['U_f_z'], W_cell=None, b=params['b_f_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_hidden = lasagne.layers.Gate(W_in=params['W_f_h'], W_hid=params['U_f_h'], W_cell=None, b=params['b_f_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_fgru_source = lasagne.layers.GRULayer(l_clookup_source, W2S_HDIM, resetgate=c2w_f_reset, updategate=c2w_f_update, hidden_update=c2w_f_hidden, hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # b-GRU
    c2w_b_reset = lasagne.layers.Gate(W_in=params['W_b_r'], W_hid=params['U_b_r'], W_cell=None, b=params['b_b_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_update = lasagne.layers.Gate(W_in=params['W_b_z'], W_hid=params['U_b_z'], W_cell=None, b=params['b_b_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_hidden = lasagne.layers.Gate(W_in=params['W_b_h'], W_hid=params['U_b_h'], W_cell=None, b=params['b_b_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_bgru_source = lasagne.layers.GRULayer(l_clookup_source, W2S_HDIM, resetgate=c2w_b_reset, updategate=c2w_b_update, hidden_update=c2w_b_hidden, hid_init=lasagne.init.Constant(0.), backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # Slice final states
    l_f_source = lasagne.layers.SliceLayer(l_fgru_source, -1, 1)
    l_b_source = lasagne.layers.SliceLayer(l_bgru_source, 0, 1)

    # Dense layer
    l_fdense_source = lasagne.layers.DenseLayer(l_f_source, SDIM, W=params['W_df'], b=params['b_df'], nonlinearity=None)
    l_bdense_source = lasagne.layers.DenseLayer(l_b_source, SDIM, W=params['W_db'], b=params['b_db'], nonlinearity=None)
    l_w2s_source = lasagne.layers.ElemwiseSumLayer([l_fdense_source, l_bdense_source], coeffs=1)

    return lasagne.layers.get_output(l_w2s_source), l_w2s_source
    
def char2word(seqs,mask,params,n_char=N_CHAR):
    '''
    Word embeddings by composing characters
    '''
    # Input layer over characters
    l_in_source = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_WORD_LENGTH,1), input_var=seqs, name='input')

    # Mask layer for variable length sequences
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_WORD_LENGTH), input_var=mask, name='mask')

    # lookup
    l_clookup_source = lasagne.layers.EmbeddingLayer(l_in_source, input_size=n_char, output_size=CHAR_DIM, W=params['Wc'])

    # f-GRU
    c2w_f_reset = lasagne.layers.Gate(W_in=params['W_f_r'], W_hid=params['U_f_r'], W_cell=None, b=params['b_f_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_update = lasagne.layers.Gate(W_in=params['W_f_z'], W_hid=params['U_f_z'], W_cell=None, b=params['b_f_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_hidden = lasagne.layers.Gate(W_in=params['W_f_h'], W_hid=params['U_f_h'], W_cell=None, b=params['b_f_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_fgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_f_reset, updategate=c2w_f_update, hidden_update=c2w_f_hidden, hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # b-GRU
    c2w_b_reset = lasagne.layers.Gate(W_in=params['W_b_r'], W_hid=params['U_b_r'], W_cell=None, b=params['b_b_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_update = lasagne.layers.Gate(W_in=params['W_b_z'], W_hid=params['U_b_z'], W_cell=None, b=params['b_b_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_hidden = lasagne.layers.Gate(W_in=params['W_b_h'], W_hid=params['U_b_h'], W_cell=None, b=params['b_b_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_bgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_b_reset, updategate=c2w_b_update, hidden_update=c2w_b_hidden, hid_init=lasagne.init.Constant(0.), backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # Slice final states
    l_f_source = lasagne.layers.SliceLayer(l_fgru_source, -1, 1)
    l_b_source = lasagne.layers.SliceLayer(l_bgru_source, 0, 1)

    # Dense layer
    l_fdense_source = lasagne.layers.DenseLayer(l_f_source, WDIM, W=params['W_df'], b=params['b_df'], nonlinearity=None)
    l_bdense_source = lasagne.layers.DenseLayer(l_b_source, WDIM, W=params['W_db'], b=params['b_db'], nonlinearity=None)
    l_c2w_source = lasagne.layers.ElemwiseSumLayer([l_fdense_source, l_bdense_source], coeffs=1)

    return lasagne.layers.get_output(l_c2w_source), l_c2w_source
    
def init_params_c2w2s(n_chars=N_CHAR):
    '''
    Initialize all params for hierarchical GRU
    '''
    params = OrderedDict()

    np.random.seed(0)

    def init_params_gru(prefix):
        gru_params = OrderedDict()

        # f-GRU
        params['W_f_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name=prefix+'W_f_r')
        params['W_f_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name=prefix+'W_f_z')
        params['W_f_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name=prefix+'W_f_h')
        params['b_f_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name=prefix+'b_f_r')
        params['b_f_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name=prefix+'b_f_z')
        params['b_f_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name=prefix+'b_f_h')
        params['U_f_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name=prefix+'U_f_r')
        params['U_f_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name=prefix+'U_f_z')
        params['U_f_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name=prefix+'U_f_h')

        # b-GRU
        params['W_b_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name=prefix+'W_b_r')
        params['W_b_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name=prefix+'W_b_z')
        params['W_b_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name=prefix+'W_b_h')
        params['b_b_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name=prefix+'b_b_r')
        params['b_b_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name=prefix+'b_b_z')
        params['b_b_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name=prefix+'b_b_h')
        params['U_b_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name=prefix+'U_b_r')
        params['U_b_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name=prefix+'U_b_z')
        params['U_b_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name=prefix+'U_b_h')

        # dense
        params['W_df'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'), name=prefix+'W_df')
        params['W_db'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'), name=prefix+'W_db')
        params['b_df'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(WDIM)).astype('float32'), name=prefix+'b_df')
        params['b_db'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(WDIM)).astype('float32'), name=prefix+'b_db')

        return gru_params

    # c2w params
    params['c2w'] = init_gru_params('c2w')
    # w2s params
    params['w2s'] = init_gru_params('w2s')
    # lookup table
    params['c2w']['Wc'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(n_chars,CHAR_DIM)).astype('float32'), name='Wc')

    return params

def tweet2vec(tweet,mask,params,n_char=N_CHAR):
    '''
    Tweet2Vec
    '''
    # Input layer over characters
    l_in_source = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_LENGTH,1), input_var=tweet, name='input')

    # Mask layer for variable length sequences
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_LENGTH), input_var=mask, name='mask')

    # lookup
    l_clookup_source = lasagne.layers.EmbeddingLayer(l_in_source, input_size=n_char, output_size=CHAR_DIM, W=params['Wc'])

    # f-GRU
    c2w_f_reset = lasagne.layers.Gate(W_in=params['W_c2w_f_r'], W_hid=params['U_c2w_f_r'], W_cell=None, b=params['b_c2w_f_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_update = lasagne.layers.Gate(W_in=params['W_c2w_f_z'], W_hid=params['U_c2w_f_z'], W_cell=None, b=params['b_c2w_f_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_hidden = lasagne.layers.Gate(W_in=params['W_c2w_f_h'], W_hid=params['U_c2w_f_h'], W_cell=None, b=params['b_c2w_f_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_fgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_f_reset, updategate=c2w_f_update, hidden_update=c2w_f_hidden, hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # b-GRU
    c2w_b_reset = lasagne.layers.Gate(W_in=params['W_c2w_b_r'], W_hid=params['U_c2w_b_r'], W_cell=None, b=params['b_c2w_b_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_update = lasagne.layers.Gate(W_in=params['W_c2w_b_z'], W_hid=params['U_c2w_b_z'], W_cell=None, b=params['b_c2w_b_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_hidden = lasagne.layers.Gate(W_in=params['W_c2w_b_h'], W_hid=params['U_c2w_b_h'], W_cell=None, b=params['b_c2w_b_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_bgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_b_reset, updategate=c2w_b_update, hidden_update=c2w_b_hidden, hid_init=lasagne.init.Constant(0.), backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # Slice final states
    l_f_source = lasagne.layers.SliceLayer(l_fgru_source, -1, 1)
    l_b_source = lasagne.layers.SliceLayer(l_bgru_source, 0, 1)

    # Dense layer
    l_fdense_source = lasagne.layers.DenseLayer(l_f_source, WDIM, W=params['W_c2w_df'], b=params['b_c2w_df'], nonlinearity=None)
    l_bdense_source = lasagne.layers.DenseLayer(l_b_source, WDIM, W=params['W_c2w_db'], b=params['b_c2w_db'], nonlinearity=None)
    l_c2w_source = lasagne.layers.ElemwiseSumLayer([l_fdense_source, l_bdense_source], coeffs=1)

    return lasagne.layers.get_output(l_c2w_source), l_c2w_source
    
def init_params(n_chars=N_CHAR):
    '''
    Initialize all params
    '''
    params = OrderedDict()

    np.random.seed(0)

    # lookup table
    params['Wc'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(n_chars,CHAR_DIM)).astype('float32'), name='Wc')

    # f-GRU
    params['W_c2w_f_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_f_r')
    params['W_c2w_f_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_f_z')
    params['W_c2w_f_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_f_h')
    params['b_c2w_f_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_f_r')
    params['b_c2w_f_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_f_z')
    params['b_c2w_f_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_f_h')
    params['U_c2w_f_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_f_r')
    params['U_c2w_f_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_f_z')
    params['U_c2w_f_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_f_h')

    # b-GRU
    params['W_c2w_b_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_b_r')
    params['W_c2w_b_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_b_z')
    params['W_c2w_b_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_b_h')
    params['b_c2w_b_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_b_r')
    params['b_c2w_b_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_b_z')
    params['b_c2w_b_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_b_h')
    params['U_c2w_b_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_b_r')
    params['U_c2w_b_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_b_z')
    params['U_c2w_b_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_b_h')

    # dense
    params['W_c2w_df'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'), name='W_c2w_df')
    params['W_c2w_db'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'), name='W_c2w_db')
    params['b_c2w_df'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(WDIM)).astype('float32'), name='b_c2w_df')
    params['b_c2w_db'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(WDIM)).astype('float32'), name='b_c2w_db')

    return params

def load_params(path):
    """
    Load previously saved model
    """
    params = OrderedDict()

    with open(path,'r') as f:
        npzfile = np.load(f)
        for kk, vv in npzfile.iteritems():
            params[kk] = vv

    return params

def load_params_shared(path):
    """
    Load previously saved model
    """
    params = OrderedDict()

    with open(path,'r') as f:
        npzfile = np.load(f)
        for kk, vv in npzfile.iteritems():
            params[kk] = theano.shared(vv, name=kk)

    return params
