import models
import numpy as np
import tensorflow as tf
from input_png import load_data
import matplotlib.pyplot as plt
from models import resnet
from test import psnr_on_y
import os
import Image

batch_size = 32
total_step = 1000
display_step = 100
learning_rate = 0.001

width = 300
height = 300


(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = load_data()
# (X_train, Y_train), (X_val, Y_val) = load_data()

"""
X_train = X_train.reshape([-1, width, height, 3])
Y_train = Y_train.reshape([-1, width, height, 3])
X_test = X_test.reshape([-1, width, height, 3])
Y_test = Y_test.reshape([-1, width, height, 3])
X_val = X_val.reshape([-1, width, height, 3])
Y_val = Y_val.reshape([-1, width, height, 3])
"""

X = tf.placeholder("float", [None, None, None, 3])
Y = tf.placeholder("float", [None, None, None, 3])

# ResNet Models
net = resnet(X, 20)
# net = models.resnet(X, 32)
# net = models.resnet(X, 44)
# net = models.resnet(X, 56)

cost = tf.reduce_mean(tf.squared_difference(net, Y))
eval_psnr = psnr_on_y(net, Y)
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
    avg_cost = 0.
    for i in range (0, len(X_train), batch_size):
        feed_dict={
            X: X_train[i:i + batch_size], 
            Y: Y_train[i:i + batch_size]}
        _, c = sess.run([train_op, cost], feed_dict=feed_dict)
        avg_cost += c/batch_size
        
        if (i+1) % 5 == 0:
            # print "training on image #%d" %(i+1)
            saver.save(sess, '/tmp/progress', global_step=i)
    	
    if (step+1) % display_step == 0:
        avg_psnr = 0
	fpath = 'cost04.txt'
        for j in range (0, len(X_val), batch_size):
	    feed_dict={X: X_val[j:j + batch_size], Y: Y_val[j:j + batch_size]}
            _, p = sess.run([train_op, eval_psnr], feed_dict=feed_dict)
            avg_psnr = p/batch_size
        print("Step: %4d"%(step+1), "Cost = {:.9f}".format(avg_cost),
                "PSNR = {:.9f}".format(avg_psnr))
	data_file = open(fpath, 'w')
	data_file.write("Cost = {:.9f}".format(avg_cost))
	data_file.write("PSNR = {:.9f}".format(avg_psnr))
	data_file.write("\n")
	data_file.close()
save_path = saver.save(sess, "mymodel04")
print("Model saved in file: %s" %save_path)

print("Optimization Finished!")


# import random
# r = random.randrange(len(X_test))
for r in range (0, len(X_test), 1):
    prediction = sess.run(net, {X: X_test[r:r+1]})
    p = psnr_on_y(prediction, Y_test[r:r+1])

    inp = X_test[r:r+1].reshape([height, width,3]).astype(np.uint8)
    outp = Y_test[r:r+1].reshape([height, width,3]).astype(np.uint8)
    prediction = prediction.reshape([height, width,3]).astype(np.uint8)

    fpath = "TestImage"
    # inp_ = tf.image.encode_png(inp)
    # outp_ = tf.image.encode_png(outp)
    # prediction_ = tf.image.encode_png(prediction)
    inp_ = Image.fromarray(inp,"RGB")
    inp_.save(os.path.join(fpath, str(i)+"input.png"))

    outp_ = Image.fromarray(outp,"RGB")
    outp_.save(os.path.join(fpath, str(i)+"output.png"))

    pred_ = Image.fromarray(prediction,"RGB")
    pred_.save(os.path.join(fpath, str(i)+"prediction.png"))

"""
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
"""
sess.close()
