import numpy as np
import tensorflow as tf

def conv2d1(x):
    with tf.variable_scope("conv2d1"):

        A0=np.zeros([21,21,3,1])
        A=np.zeros([21,21,1])
        for i in range(20):
            A[i,i,:]=-1  
            A[20-i,20-i,:]=-1  
            A[i,20-i,:]=1  
            A[20-i,i,:]=1   
        A[10,10,:]=0    
        for ii in range(3):
            A0[:,:,ii,:] = A
        W1=tf.convert_to_tensor(A0,dtype='float32') 
        return tf.nn.conv2d(x, W1, strides=[1, 2, 2, 1], padding='SAME') 
