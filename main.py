import models
import numpy as np
import tensorflow as tf
from input_bsds500 import load_data
import matplotlib.pyplot as plt

def psnr(target, ref):
	diff = ref - target
	diff = tf.reshape(diff, [tf.size(diff)])
	rmse = tf.sqrt( tf.reduce_mean(diff ** 2.) )
	return 20*tf.log(1.0/rmse)/tf.log(tf.constant(10.))

batch_size = 64
total_step = 50
display_step = 10
learning_rate = 0.001

width = 50
height = 50

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', learning_rate, 'Learning rate')
flags.DEFINE_integer('batch_size', batch_size, 'Batch size')

(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = load_data()
trainX = trainX.reshape([-1, width, height, 3])
trainY = trainY.reshape([-1, width, height, 3])
testX = testX.reshape([-1, width, height, 3])
testY = testY.reshape([-1, width, height, 3])
valX = valX.reshape([-1, width, height, 3])
valY = valY.reshape([-1, width, height, 3])

X = tf.placeholder("float", [batch_size, 50, 50, 3])
Y = tf.placeholder("float", [batch_size, 50, 50, 3])

# ResNet Models
net = models.resnet(X, 20)
# net = models.resnet(X, 32)
# net = models.resnet(X, 44)
# net = models.resnet(X, 56)

cost = tf.reduce_mean(tf.squared_difference(net, Y))
eval_psnr = psnr(net, Y)
train_op = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint(".")
if checkpoint:
    print "Restoring from checkpoint", checkpoint
    saver.restore(sess, checkpoint)
else:
    print "Couldn't find checkpoint to restore from. Starting over."

for step in range (total_step):
    for i in range (0, len(X_train), batch_size):
        feed_dict={
            X: X_train[i:i + batch_size], 
            Y: Y_train[i:i + batch_size],
            learning_rate: learning_rate}
        _, c = sess.run([train_op, cost], feed_dict=feed_dict)
        if i % 30 == 0:
            print "training on image #%d" % i
            saver.save(sess, 'progress', global_step=i)

    if (step+1) % 4 ==0:
        for j in range(0, len(X_val), batch_size):
            feed_dict={
                X: X_val[i:i + batch_size], 
                Y: Y_val[i:i + batch_size],
                learning_rate: learning_rate}
            _, p = sess.run([train_op, eval_psnr], feed_dict=feed_dict)
    if (step+1) % display_step == 0:
        print("Step: ", '%4d'%(step+1), "Cost = ", "{:.9f}".format(cost),
                "PSNR =", "{:.9f}".format(psnr))

print("Optimization Finished!")

import random
r = random.randrange(len(testX))
prediction = sess.run(Y_, {X: testX[r:r+1]})
a = fig.add_subplot(1,3,1)
a.set_title('Noise Image(Input)')
plt.imshow(testX[r:r+1])
b = fig.add_subplot(1,3,2)
b.set_title('Denoise Image(Output)')
b.set_xlabel("PSNR = ", "{:.9f}".format(psnr(prediction, testY[r:r+1])),
             "SSIM = ", "{:.9f}".format(ssim(prediction, testY[r:r+1])))
plt.imshow(prediction)
c = fig.add_subplot(1,3,3)
c.set_title('Clean Image(Compare)')
plt.imshow(testY[r:r+1])

fig.suptitle('Random Test')
plt.show()

sess.close()

    
    
