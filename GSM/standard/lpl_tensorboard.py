'''lpl 19/06/04 21:39
主要用于对数据进行可视化，使用的工具时tensorboard'''
import tensorflow as tf
'''对一个2D或1D变量进行平均值，最大值，方差等数据特征进行可视化'''
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)
def variable4D_summaries(var4D):
    shape=var4D.shape
    var4D=tf.reshape(var4D,shape=[-1,2])
    with tf.name_scope('pro'):
        max_index=tf.argmax(var4D,1)
        obj_num=tf.reduce_sum(max_index)
        tf.summary.scalar('obj_num',obj_num)
        #tf.summary.scalar('bk_num', tf.subtract(tf.constant(24*24,dtype=tf.int64),obj_num))
        tf.summary.scalar('bk_num', tf.subtract(tf.multiply(shape[1],shape[2]), tf.cast(obj_num,dtype=tf.int32)))
        mean_v=tf.reduce_mean(var4D)
        tf.summary.scalar('mean',mean_v)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var4D - mean_v)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram',var4D)

def variable4D_summaries1(var4D,map_width,map_hight):
    var4D = tf.reshape(var4D, shape=[-1, 2])
    with tf.name_scope('pro'):
        max_index = tf.argmax(var4D, 1)

        obj_num = tf.reduce_sum(max_index)
        tf.summary.scalar('obj_num', obj_num)
        tf.summary.scalar('bk_num', tf.subtract(tf.constant(map_width * map_hight, dtype=tf.int64), obj_num))
        mean_v = tf.reduce_mean(var4D)
        tf.summary.scalar('mean', mean_v)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var4D - mean_v)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var4D)
#可视化所有梯度
def gradient_summaries(gradients):
    with tf.name_scope("gradient"):
        for g, v in gradients:
            square_v=tf.square(v)
            sum_v=tf.reduce_sum(square_v)
            tf.summary.scalar(v.name,sum_v)
