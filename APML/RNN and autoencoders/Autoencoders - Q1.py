import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA as PCA
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



# Training Parameters
learning_rate = 0.01
num_steps = 5000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
dim_hidden = [100,2] # num of features for 2 layers according to i
dim_input = 784 # MNIST data input (img shape: 28*28)
num_of_encoder_layers = len(dim_hidden)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, dim_input])

# weights and biases initialization for the encoder and decoder
encoder_bias = np.zeros((num_of_encoder_layers))
decoder_bias = np.zeros((num_of_encoder_layers))

encoder_weights = [tf.Variable(tf.random_normal([dim_input, dim_hidden[0]])),
                   tf.Variable(tf.random_normal([dim_hidden[0], dim_hidden[1]]))]

decoder_weights = [tf.Variable(tf.random_normal([dim_hidden[1], dim_hidden[0]])),
                   tf.Variable(tf.random_normal([dim_hidden[0], dim_input]))]


encoder_bias = [tf.Variable(tf.random_normal([dim_hidden[0]])),
                tf.Variable(tf.random_normal([dim_hidden[1]]))]


decoder_bias = [tf.Variable(tf.random_normal([dim_hidden[0]])),
                tf.Variable(tf.random_normal([dim_input]))]


# Building the encoder
def encoder(x):

    # Encoder Hidden layer with sigmoid activation #1 until #8
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_weights[0]), encoder_bias[0]))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, encoder_weights[1]), encoder_bias[1]))

    return layer_2


# Building the decoder
def decoder(x):

    # decoder Hidden layer with sigmoid activation #1 until #8
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_weights[0]), decoder_bias[0]))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, decoder_weights[1]), decoder_bias[1]))

    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op

#dim reduction
y_reduced = encoder_op

# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    size_of_arr = int(num_steps/display_step)
    train_loss = np.zeros((num_steps))

    for i in range(num_steps):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        tmp, train_loss[i] = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, train_loss[i]))

    # Training loss
    plt.figure(figsize=(10, 10))
    plt.plot(list(range(0, num_steps)), train_loss, label='Train Loss')
    plt.title('Train Loss -  iter num %d batch size %d leaning rate %f' % (num_steps, batch_size, learning_rate))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(
        'Test train - iter num %d batch size %d leaning rate %f_autoencoders.png' % (num_steps, batch_size, learning_rate))
    plt.show()

    n = 1000
    test_loss = np.zeros((n))
    accuracy_iter = np.zeros((n))

    for i in range(n):
        # MNIST test set
        batch_x, labels = mnist.test.next_batch(n)
        labels = np.argmax(labels, axis = 1)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})
        test_loss[i] = sess.run(loss, feed_dict={X: batch_x})
        g_only_encoder = sess.run(encoder_op, feed_dict={X: batch_x})



    # test loss
    plt.figure(figsize=(10, 10))
    plt.plot(list(range(0, num_steps)), train_loss, label='Train Loss')
    plt.title('Test Loss -  iter num %d batch size %d leaning rate %f' % (num_steps, batch_size, learning_rate))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(
        'Test Loss - iter num %d batch size %d leaning rate %f_autoencoders.png' % (num_steps, batch_size, learning_rate))
    plt.show()

    # plt.figure(figsize=(10, 10))
    # plt.plot(list(range(0, num_steps)), accuracy, label='Accuracy')
    # plt.title('Accuracy Loss -  iter num %d batch size %d leaning rate %f' % (num_steps, batch_size, learning_rate))
    # plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    # plt.savefig(
    #     ' accuracy - iter num %d batch size %d leaning rate %f_autoencoders.png' % (
    #         num_steps, batch_size, learning_rate))
    # plt.show()

    # 2 dim after encoder
    plt.figure(figsize=(10, 10))
    plt.scatter(np.asarray(g_only_encoder)[:, 0], np.asarray(g_only_encoder)[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.title("2- Dimensional Embedding of the MNIST  - using Autoencoders")
    plt.savefig("red_iter_auto%d.png" % num_steps)
    plt.show()

    # 2 dim after PCA
    plt.figure(figsize=(10, 10))
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(batch_x)
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.title("2- Dimensional Embedding of the MNIST - using PCA")
    plt.savefig("red_iter_PCA%d.png" % num_steps)
    plt.show()
