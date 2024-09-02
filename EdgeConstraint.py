import numpy as np
import tensorflow as tf

def conv2d1(x):
    with tf.variable_scope("conv2d1"):
        #W1 = tf.get_variable("W1", [10, 10, 3, 1], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        #W1=tf.ones([10,10,3,1])
        A0=np.zeros([21,21,3,1])#构造一个21*21的卷积核，两条对角线上值分别为全1，全-1，中心及其他地方值为0
        A=np.zeros([21,21,1])
        for i in range(20):
            A[i,i,:]=-1  #左上角的半条对角线值为-1
            A[20-i,20-i,:]=-1  #右小角的半条对角线值为-1
            A[i,20-i,:]=1  #右上角的半条对焦线上值为1
            A[20-i,i,:]=1   #左下角的半条对焦线上值为1
        A[10,10,:]=0    #中心点处值为0
        for ii in range(3):
            A0[:,:,ii,:] = A
        W1=tf.convert_to_tensor(A0,dtype='float32') #将卷积核从np矩阵格式转换为tensorflow可以处理的tensor格式
        #print (A0)
        return tf.nn.conv2d(x, W1, strides=[1, 2, 2, 1], padding='SAME') #返回结果为用我们定义的卷积核实现的二维卷积
