import models
import numpy as np
import tensorflow as tf
from input_bsds500 import load_data
import matplotlib.pyplot as plt
from models import resnet

def psnr(target, ref):
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(target, ref)))
    result = 20*tf.log(256*256*3/rmse)/tf.log(tf.constant(10.))
    return result

batch_size = 32
total_step = 50
display_step = 10
learning_rate = 0.001

width = 50
height = 50

"""
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', learning_rate, 'Learning rate')
flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
"""

(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = load_data()
"""
X_train = X_train.reshape([-1, width, height, 3])
Y_train = Y_train.reshape([-1, width, height, 3])
X_test = X_test.reshape([-1, width, height, 3])
Y_test = Y_test.reshape([-1, width, height, 3])
X_val = X_val.reshape([-1, width, height, 3])
Y_val = Y_val.reshape([-1, width, height, 3])
"""

X = tf.placeholder("float", [None, 50, 50, 3])
Y = tf.placeholder("float", [None, 50, 50, 3])

# ResNet Models
net = resnet(X, 20)
# net = models.resnet(X, 32)
# net = models.resnet(X, 44)
# net = models.resnet(X, 56)

cost = tf.reduce_mean(tf.squared_difference(net, Y))
eval_psnr = psnr(net, Y)
train_op = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
"""
saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint(".")
if checkpoint:
    print "Restoring from checkpoint", checkpoint
    saver.restore(sess, checkpoint)
else:
    print "Couldn't find checkpoint to restore from. Starting over."
"""

for step in range (total_step):
    avg_cost = 0.
    for i in range (0, len(X_train), batch_size):
        feed_dict={
            X: X_train[i:i + batch_size], 
            Y: Y_train[i:i + batch_size]}
        _, c = sess.run([train_op, cost], feed_dict=feed_dict)
        avg_cost += c/batch_size
        
        if (i+1) % 5 == 0:
            print "training on image #%d" %(i+1)
            # saver.save(sess, 'progress', global_step=i)
    
    if (step+1) % display_step == 0:
        for j in range(0, len(X_val), batch_size):
            feed_dict={
                X: X_val[j:j + batch_size], 
                Y: Y_val[j:j + batch_size]}
            _, p = sess.run([train_op, eval_psnr], feed_dict=feed_dict)
            avg_psnr = p/batch_size
        print("Step: %4d"%(step+1), "Cost = {:.9f}".format(avg_cost),
                "PSNR = {:.9f}".format(avg_psnr))

print("Optimization Finished!")


import random
r = random.randrange(len(X_test))
prediction = sess.run(net, {X: X_test[r:r+1]})
p = format(psnr(prediction, Y_test[r:r+1]))

inp = X_test[r:r+1].reshape([width, height, 3]).astype(np.uint8)
outp = Y_test[r:r+1].reshape([width, height, 3]).astype(np.uint8)
prediction = prediction.reshape([width, height, 3]).astype(np.uint8)
fig = plt.figure()
a = fig.add_subplot(1,3,1)
a.set_title('Noise Image(Input)')
plt.imshow(inp)
b = fig.add_subplot(1,3,2)
b.set_title('Denoise Image(Output)')
_psnr = "PSNR = "+ p
b.set_xlabel(_psnr)
plt.imshow(prediction)
c = fig.add_subplot(1,3,3)
c.set_title('Clean Image(Compare)')
plt.imshow(outp)

fig.suptitle('Random Test')
plt.show()

sess.close()
