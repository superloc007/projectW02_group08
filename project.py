import os
from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# the data folder
data_dir = "data"
# if train = false, the program will run the test mode. if train = true, the program will run the train mode
train = False
model_path = "model/image_model"
def read_data(data_dir):
    datas = []
    fpaths = []
    labels = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        images = Image.open(fpath).resize((32,32))
        data = np.array(images) / 255.0
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)
    datas = np.array(datas)
    labels = np.array(labels)
    return fpaths, datas, labels
fpaths, datas, labels = read_data(data_dir)
#Count how many types of pictures there are
num_classes = len(set(labels))
#Define the place holder, store the input and the labels
datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
dropout_placeholder = tf.placeholder(tf.float32)
labels_placeholder = tf.placeholder(tf.int32, [None])
# define the convention layer
conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
# define the pooling layer
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
#transfer the 3D arrays to 1D
flatten = tf.layers.flatten(pool1)
# the fully connecting layer
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)
# avoid overfitting
dropout_fc = tf.layers.dropout(fc, dropout_placeholder)
logits = tf.layers.dense(dropout_fc, num_classes)
predicted_labels = tf.arg_max(logits, 1)
#Loss is defined by cross entropy
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)
# the mean loss
mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)
saver = tf.train.Saver()

with tf.Session() as sess:
    if train:
        print("begin the train mode")
        sess.run(tf.global_variables_initializer())
        train_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholder: 0.25
        }
        for step in range(150):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step, mean_loss_val))
        saver.save(sess, model_path)
        print("finish training")
    else:
        print("begin the test mode")
        saver.restore(sess, model_path)
        print("load the model from{}".format(model_path))
        label_name_dict = {
            0: "plane",
            1: "car",
            2: "birds"
        }
        test_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholder: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            print(fpath+" : "+predicted_label_name)










