import numpy as np
import torch


def MinMaxScaler(dataX):
    
    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    dataX = dataX - min_val
    
    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    dataX = dataX / (max_val + 1e-7)
    
    return dataX, min_val, max_val


#%% Start TGAN function (Input: Original data, Output: Synthetic Data)
def tgan (dataX, parameters):
	# tf.reset_default_graph()  # https://discuss.pytorch.org/t/how-to-free-graph-manually/9255

	# Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])

    # Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))

    # Normalization
    if ((np.max(dataX) > 1) | (np.min(dataX) < 0)):
        dataX, min_val, max_val = MinMaxScaler(dataX)
        Normalization_Flag = 1
    else:
        Normalization_Flag = 0
     
    # Network Parameters
    hidden_dim   = parameters['hidden_dim'] 
    num_layers   = parameters['num_layers']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module_name']    # 'lstm' or 'lstmLN'
    z_dim        = parameters['z_dim']
    gamma        = 1
    
    #%% input place holders  # https://discuss.pytorch.org/t/placeholder-in-pytorch/96614
    
    # X = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x")
    # Z = tf.placeholder(tf.float32, [None, Max_Seq_Len, z_dim], name = "myinput_z")
    # T = tf.placeholder(tf.int32, [None], name = "myinput_t")

    # def rnn_cell  # class TimeGAN(Module):

