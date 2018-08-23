import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 100
neu = 256


X = tf.placeholder(tf.float32, [None, 7])
Y = tf.placeholder(tf.float32, [None, 1])

# weights & bias for nn layers
W1 = tf.get_variable("W1", shape=[7, neu])
b1 = tf.Variable(tf.random_normal([neu]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[neu, neu])
b2 = tf.Variable(tf.random_normal([neu]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[neu, neu])
b3 = tf.Variable(tf.random_normal([neu]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable("W4", shape=[neu, neu])
b4 = tf.Variable(tf.random_normal([neu]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

W5 = tf.get_variable("W5", shape=[neu, 1])
b5 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L4, W5) + b5

sess = tf.Session()
saver = tf.train.Saver()

saver.restore(sess, './learned_model/nn_epoch-500_Lrate-0.001_neu-256_trackdata_p_fb_degout_obin_deg')


