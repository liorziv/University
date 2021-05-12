import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA as PCA
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def interpolate_pics(pics, interpolation_factor):
    a = pics[0,:] * interpolation_factor + (1 - interpolation_factor) * pics[1,:]


    return a

# Training Parameters
learning_rate = 0.01
num_steps = 10000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
dim_hidden  = [100,25,2]  # num of features for 8 layers according to i
dim_input = 784 # MNIST data input (img shape: 28*28)
num_of_encoder_layers = len(dim_hidden)
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, dim_input])

# weights and biases initialization for the encoder and decoder

encoder_bias = np.zeros((num_of_encoder_layers))
decoder_bias = np.zeros((num_of_encoder_layers))


encoder_weights = [tf.Variable(tf.random_normal([dim_input, dim_hidden[0]])),
                   tf.Variable(tf.random_normal([dim_hidden[0], dim_hidden[1]])),
                   tf.Variable(tf.random_normal([dim_hidden[1], dim_hidden[2]]))]





decoder_weights = [tf.Variable(tf.random_normal([dim_hidden[2], dim_hidden[1]])),
                   tf.Variable(tf.random_normal([dim_hidden[1], dim_hidden[0]])),
                   tf.Variable(tf.random_normal([dim_hidden[0], dim_input]))]


encoder_bias = [tf.Variable(tf.random_normal([dim_hidden[0]])),
                tf.Variable(tf.random_normal([dim_hidden[1]])),
                tf.Variable(tf.random_normal([dim_hidden[2]]))]


decoder_bias = [tf.Variable(tf.random_normal([dim_hidden[1]])),
                tf.Variable(tf.random_normal([dim_hidden[0]])),
                tf.Variable(tf.random_normal([dim_input]))]



# Building the encoder
def encoder(x):

    # Encoder Hidden layer with sigmoid activation #1 until #8
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_weights[0]), encoder_bias[0]))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, encoder_weights[1]), encoder_bias[1]))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, encoder_weights[2]), encoder_bias[2]))


    return layer_3


# Building the decoder
def decoder(x):

    # decoder Hidden layer with sigmoid activation #1 until #8
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_weights[0]), decoder_bias[0]))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, decoder_weights[1]), decoder_bias[1]))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, decoder_weights[2]), decoder_bias[2]))


    return layer_3

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
        # Get the next batch of MNIST data
        batch_x, _ = mnist.train.next_batch(batch_size)
        #train the model
        tmp, train_loss[i] = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, train_loss[i]))

    n = 2
    size = int(n*2)
    canvas_recon = np.empty((28 * size , 28 * size))
    orig_pics =  np.empty((28 * n , 28 * n))

    g = []
    # MNIST test set
    batch_x, labels = mnist.test.next_batch(n)
    # test_loss[i] = sess.run(loss, feed_dict={X: batch_x})
    g_latent_space = sess.run(encoder_op, feed_dict={X: batch_x})
    #Display original images
    canvas_interpolation = interpolate_pics(g_latent_space,0.1)
    g.append(sess.run(decoder_op, feed_dict={encoder_op: canvas_interpolation.reshape(1, 2)}))

    canvas_interpolation = interpolate_pics(g_latent_space, 0.3)
    g.append(sess.run(decoder_op, feed_dict={encoder_op: canvas_interpolation.reshape(1, 2)}))

    canvas_interpolation = interpolate_pics(g_latent_space, 0.6)
    g.append(sess.run(decoder_op, feed_dict={encoder_op: canvas_interpolation.reshape(1, 2)}))

    canvas_interpolation = interpolate_pics(g_latent_space, 0.9)
    g.append(sess.run(decoder_op, feed_dict={encoder_op: canvas_interpolation.reshape(1, 2)}))




    # Display reconstructed images
    for j in range(4):
        # Draw the reconstructed digits
        canvas_recon[:  1 * 28, j * 28:(j + 1) * 28] = \
        np.array(g[j]).reshape([28, 28])
        # Display original images
    for j in range(n):
        orig_pics[: 1 * 28, j * 28:(j + 1) * 28] = \
            batch_x[j].reshape([28, 28])




    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.savefig("recon_iter_laten%d.png" % num_steps)
    plt.show()

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(orig_pics, origin="upper", cmap="gray")
    plt.savefig("orig_iter_laten%d.png" % num_steps)
    plt.show()

    plt.figure(figsize=(10, 10))
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(batch_x)
    g_pca = []

    # interpolation using pca
    canvas_interpolation_pca = interpolate_pics(x_reduced, 0.1)
    g.append(pca.inverse_transform(canvas_interpolation_pca))

    canvas_interpolation_pca = interpolate_pics(x_reduced, 0.3)
    g.append(pca.inverse_transform(canvas_interpolation_pca))

    canvas_interpolation_pca = interpolate_pics(x_reduced, 0.6)
    g.append(pca.inverse_transform(canvas_interpolation_pca))

    canvas_interpolation_pca = interpolate_pics(x_reduced, 0.9)
    g.append(pca.inverse_transform(canvas_interpolation_pca))

    canvas_recon_pca = np.empty((28 * size, 28 * size))
    for j in range(4):
        # Draw the reconstructed digits
        canvas_recon_pca[:  1 * 28, j * 28:(j + 1) * 28] = \
            np.array(g[j]).reshape([28, 28])


    print("pca Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon_pca, origin="upper", cmap="gray")
    plt.savefig("re_iter_pca%d.png" % num_steps)
    plt.show()


