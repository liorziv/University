from scipy import signal
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# define the functions we would like to predict:
num_of_functions = 3
size = 4
W = 4 * (np.random.random((size, size)) - 0.5)
y = {
    0: lambda x: np.sum(np.dot(x, W), axis=1),
    1: lambda x: (x[:, 0] * W[0, 0] + x[:, 1] * W[1, 1]) * (
        x[:, 2] * W[2, 2] + x[:, 1] * W[3, 3]),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x, W)), axis=1))
}

def mean_square_loss(p,y):
    '''
    calculates square loss
    :param p: the prediction (wx, batch of x)
    :param y: the training/test labels
    :return: mean of least square loss
    '''
    return np.mean((p - y) ** 2)

def mean_square_loss_reg(p,y,lamb,w):
    '''
    calculates square loss
    :param p: the prediction (wx, batch of x)
    :param y: the training/test labels
    :param lamb: the regularization parameter
    :param w: the weights matrix
    :return: mean of least square loss
    '''
    return np.mean((p - y) ** 2 + (lamb / 2)) * (w.dot(w))

def learn_linear(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a linear model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (w, training_loss, test_loss):
            w: the weights of the linear model
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    w = {func_id: np.zeros(size) for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):
        for _ in range(iterations):

            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx,:], Y[func_id]['train'][idx]

            # calculate the loss and derivatives:
            p = np.dot(x, w[func_id])
            loss = mean_square_loss_reg(p,y,lamb,w[func_id])
            test_p =  X['test'].dot(w[func_id])

            iteration_test_loss = mean_square_loss_reg(test_p,Y[func_id]['test'],lamb,w[func_id])
            # dl_dw = np.zeros(size)
            # for i in range(size):
            #
            #     dl_dw[i] = np.mean((2*((p - y)*x[:,i])))+lamb*w[func_id][i]
            #
            # update the model and record the loss:
            dl_dw = np.mean(np.multiply(x.T, 2 * (p - y)).T, axis=0) + (lamb * w[func_id])

            w[func_id] -= learning_rate * dl_dw
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)


    return w, training_loss, test_loss

def test_linear_leaner(X,Y,batch_size, lamb, iterations, learning_rate ):
    '''

    Test a linear model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    '''
    w,training_loss,test_loss = learn_linear(X, Y,batch_size, lamb, iterations, learning_rate )
    plt.figure(figsize=(10, 10))

    for i in range(num_of_functions):

        plt.plot(list(range(iterations)),training_loss[i],label = 'func%d'%i)

    plt.title('Test Loss for batch size %d and lambda %f' %(batch_size,lamb))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Test_loss_%d_%f.png' %(batch_size,lamb))
    plt.show()

def forward(cnn_model, x):
    """
    Given the CNN model, fill up a dictionary with the forward pass values.
    :param cnn_model: the model
    :param x: the input of the CNN
    :return: a dictionary with the forward pass values
    """

    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [np.array(cnn_model['w1'])], mode='same'))
    fwd['o2'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [cnn_model['w2']], mode='same'))
    fwd['m1'] = np.zeros((size(x)))
    fwd['m'] = [np.maximum(fwd['o1'][:,:2]),np.maximum(fwd['o1'][:,3:]), np.maximum(fwd['o2'][:,:2]), np.maximum(fwd['o2'][:,3:])]
    fwd['m_argmax'] = [np.argmax(fwd['o1'][0:2]), np.argmax(fwd['o1'][3:]),np.argmax(fwd['o2'][0:2]), np.argmax(fwd['o2'][3:])]
    fwd['p'] = cnn_model['u'].dot(fwd['m'].T)

    return fwd

def backprop(model, y, fwd, batch_size):
    """
    given the forward pass values and the labels, calculate the derivatives
    using the back propagation algorithm.
    :param model: the model
    :param y: the labels
    :param fwd: the forward pass values
    :param batch_size: the batch size
    :return: a tuple of (dl_dw1, dl_dw2, dl_du)
            dl_dw1: the derivative of the w1 vector
            dl_dw2: the derivative of the w2 vector
            dl_du: the derivative of the u vector
    """
    p_w1 = np.dot(fwd['x'],model['w1'])
    p_w2 = np.dot(fwd['x'],model['w2'])

    u_1_3 = [fwd['u'][1],fwd['u'][3]]
    m_by_o = np.array([0,1,0,0],[0,0,])

    dl_dw1 = np.mean(np.multiply(x, 2 * (p_w1 - y)).T, axis=0)

    dl_dw2 = np.mean(np.multiply(fwd['x'], 2 * (p_w2 - y)).T, axis=0)

    dl_du =  np.mean(np.multiply(model['m'].T, 2 * (fwd['p'] - y)).T, axis=0)

    return (dl_dw1, dl_dw2, dl_du)

def learn_cnn(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a cnn model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (models, training_loss, test_loss):
            models: a model for every function (a dictionary for the parameters)
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    models = {func_id: {} for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):

        # initialize the model:
        models[func_id]['w1'] = np.array([1]*4)
        models[func_id]['w2'] = np.array([1]*4)
        models[func_id]['u'] = np.array([1]*4)

        # train the network:
        for _ in range(iterations):

            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx,:], Y[func_id]['train'][idx]

            # calculate the loss and derivatives using back propagation:
            fwd = forward(models[func_id], x)
            loss = mean_square_loss(fwd['p'], y)
            dl_dw1, dl_dw2, dl_du = backprop(models[func_id], y, fwd, batch_size)

            # record the test loss before updating the model:
            test_fwd = forward(models[func_id], X['test'])
            iteration_test_loss = mean_square_loss(test_fwd['p'], y)

            # update the model using the derivatives and record the loss:
            models[func_id]['w1'] -= learning_rate * dl_dw1
            models[func_id]['w2'] -= learning_rate * dl_dw2
            models[func_id]['u'] -= learning_rate * dl_du
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return models, training_loss, test_loss

def test_cnn(X, Y, batch_size, lamb, iterations, learning_rate):
    learn_cnn(X,Y,batch_size, lamb, iterations, learning_rate)

def print_linear_tensorflow_by_batch(amount_of_samples, train_loss,test_loss,accuracy_iter, iter_num, batch,learning_rate,acc):



        plt.figure(figsize=(10, 10))


        for i in range(amount_of_samples):
            plt.plot(list(range(0,iter_num,100)), accuracy_iter[i], label='batch size %d'%batch[i])


        plt.legend()
        plt.title('Test accuracy -  num of iteration %d  learning rate %f result(%f)' %(iter_num, learning_rate, acc[i]))
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig( 'Test accuracy -  num of iteration %d  learning rate %f _first_deep_comp.png' %(iter_num, learning_rate))

        plt.show()


        for i in range(amount_of_samples):
            plt.plot(list(range(iter_num)), train_loss[i], label='Train - batch size %d'%batch[i])



        plt.title('Training  Loss -  num of iteration %d learning rate %f' % (iter_num, learning_rate))
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Training Loss -  num of iteration %d  learning rate %f  _first_deep_comp.png' % (iter_num, learning_rate))
        plt.show()

        for i in range(amount_of_samples):

            plt.plot(list(range(0, iter_num, 100)), test_loss[i], label='Test - batch size %d' % batch[i])

        plt.title('Test Loss -  num of iteration %d learning rate %f' % (iter_num, learning_rate))
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Test Loss -  num of iteration %d  learning rate %f  _first_deep_comp.png' % (
        iter_num, learning_rate))
        plt.show()

def linear_tensorflow(iter_num, batch_size, learning_rate):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # open a session
    sess = tf.InteractiveSession()

    # define input and output
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # define the weights and bias
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    #inialize the parameters
    sess.run(tf.global_variables_initializer())

    #building our regression model
    y = tf.matmul(x, W) + b

    # define a loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    #training starts
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    tf.summary.scalar('loss', cross_entropy)
    size_of_arr = int(iter_num/100)
    accuracy_iter = np.zeros(size_of_arr)
    train_loss = np.zeros(int(iter_num))
    test_loss = np.zeros(size_of_arr)
    for _ in range(iter_num):
        batch = mnist.train.next_batch(batch_size)
        tmp, train_loss[_] = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1]})
        if (_ % 100 == 0):
            curr_place = int(_/100)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_iter[curr_place] = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            test_loss[curr_place] = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print(_)


    # Evluating the results
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    plt.figure(figsize=(10, 10))
    plt.plot(list(range(int(iter_num/100))), accuracy_iter, label='Accuracy')
    plt.title('Test accuracy - iter num %d batch size %d leaning rate %f' % (iter_num, batch_size, learning_rate))
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('Test accuracy - iter num %d batch size %d leaning rate %f_linear_tn.png' % (iter_num, batch_size, learning_rate))
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(list(range(iter_num)), train_loss, label='Training Loss')
    plt.title('Training_Test Loss  iter num %d batch size %d leaning rate %f' % (iter_num, batch_size, learning_rate))
    plt.plot(list(range(0, iter_num, 100)), test_loss, label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Training_Test Loss  iter num %d batch size %d leaning rate %f_linear_tn.png' % (iter_num, batch_size, learning_rate))
    plt.show()





def first_deep_network(iter_num, batch_size, learning_rate):


    #creating a model for multilayer preceptron


    hidden_1 = 256  # 1st layer number of neurons
    hidden_2 = 256  # 2nd layer number of neurons
    input = 784  # MNIST data input (img shape: 28*28)
    output = 10  # MNIST total classes (0-9 digits)
    h_1 = 0
    h_2 = 1
    out = 2

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # open a session
    sess = tf.InteractiveSession()

    # define input and output
    x = tf.placeholder(tf.float32, shape=[None, input])
    y_ = tf.placeholder(tf.float32, shape=[None, output])


    # Store layers weight & bias
    weights = [tf.Variable(tf.random_normal([input, hidden_1])),tf.Variable(tf.random_normal([hidden_1, hidden_2])),
        tf.Variable(tf.random_normal([hidden_2, output]))]

    bias = [tf.Variable(tf.random_normal([hidden_1])),tf.Variable(tf.random_normal([hidden_2])),
    tf.Variable(tf.random_normal([output]))]

    # inialize the parameters
    sess.run(tf.global_variables_initializer())

    layer_1 = tf.nn.relu(tf.matmul(x, weights[h_1])+ bias[h_1])
    layer_2 =  tf.add(tf.matmul(layer_1, weights[h_2]) , bias[h_2])
    y = tf.matmul(layer_2, weights[out]) + bias[out]

    # define a loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # training starts
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    tf.summary.scalar('loss', cross_entropy)
    size_of_arr = int(iter_num / 100)
    accuracy_iter = np.zeros((size_of_arr))

    train_loss = np.zeros((iter_num))
    test_loss = np.zeros((size_of_arr))
    for _ in range(iter_num):
        print(_)

        batch = mnist.train.next_batch(batch_size)
        tmp, train_loss[_] = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1]})
        if(_%100 == 0):
            curr_idx = int(_ / 100)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_iter[curr_idx] = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            test_loss[curr_idx] = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print(_)

    acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('accuracy of %f'%acc)

    plt.figure(figsize=(10, 10))
    plt.plot(list(range(0,iter_num,100)), accuracy_iter, label='Accuracy')
    plt.title('Test accuracy -  iter num %d batch size %d leaning rate %f' % (iter_num, batch_size, learning_rate))
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig(
        'Test accuracy - iter num %d batch size %d leaning rate %f_First_D_n.png' % (iter_num, batch_size, learning_rate))
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(list(range(iter_num)), train_loss, label='Train Loss')
    plt.title('Training_Test Loss  iter num %d batch size %d leaning rate %f' % (iter_num, batch_size, learning_rate))
    plt.plot(list(range(0,iter_num,100)), test_loss, label='Test Loss')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Training_Test Loss  iter num %d batch size %d leaning rate %f_First_D_n.png' % (iter_num, batch_size, learning_rate))
    plt.show()

    return train_loss, test_loss, accuracy_iter,acc

def conv_neural_net(iter_num, batch_size, learning_rate):

    hidden_1 = 256  # 1st layer number of neurons
    hidden_2 = 256  # 2nd layer number of neurons
    input = 784  # MNIST data input (img shape: 28*28)
    output = 10  # MNIST total classes (0-9 digits)
    h_1 = 0
    h_2 = 1
    out = 2

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # open a session
    sess = tf.InteractiveSession()

    # define input and output
    x = tf.placeholder(tf.float32, shape=[None, input])
    y_ = tf.placeholder(tf.float32, shape=[None, output])

    # create weights vector
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # create bias vector
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    # convultion layer
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # pooling layer
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    #first layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #densly connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Dropout

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # train
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step =  tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', cross_entropy)
    size_of_arr = int(iter_num/100)
    accuracy_iter = np.zeros((size_of_arr))
    train_loss = np.zeros((iter_num))
    test_loss = np.zeros((size_of_arr))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iter_num):

            batch = mnist.train.next_batch(batch_size)
            tmp, train_loss[i] = sess.run([train_step, cross_entropy],
                                          feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if i % 100 == 0:
                print(i)
                curr_idx = int(i / 100)
                accuracy_iter[curr_idx] = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                test_loss[curr_idx] = sess.run(cross_entropy, feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})




        plt.figure(figsize=(10, 10))
        plt.plot(list(range(0,iter_num,100)), accuracy_iter, label='Accuracy')
        plt.title('Test accuracy - iter num %d batch size %d learning rate %f' % (iter_num, batch_size, learning_rate))
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Test accuracy - iter num %d batch size %d leaning rate %f_conv_NN.png' % (iter_num, batch_size, learning_rate))
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.plot(list(range(iter_num)), train_loss, label='Training Loss')
        plt.title('Training_Test Loss  iter num %d batch size %d learning rate %f' % (iter_num, batch_size, learning_rate))
        plt.plot(list(range(0,iter_num,100)), test_loss, label='Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Training_Test - iter num %d batch size %d learning rate %f_conv_NN.png' % (iter_num, batch_size, learning_rate))
        plt.show()

if __name__ == '__main__':

    for i in range(10):
        print(i)
    # generate the training and test data, adding some noise:
    X = dict(train=5 * (np.random.random((1000, size)) - .5),
             test=5 * (np.random.random((200, size)) - .5))
    Y = {i: {
        'train': y[i](X['train']) * (
        1 + np.random.randn(X['train'].shape[0]) * .01),
        'test': y[i](X['test']) * (
        1 + np.random.randn(X['test'].shape[0]) * .01)}
         for i in range(len(y))}

    #test_linear_leaner(X, Y, 50 , 0.5, 3500, 0.001)
    #test_cnn(X, Y, 10 , 0.5, 3500, 0.01)
 # *************************** checking linear tensor flow ***************************

    # linear_tensorflow(iter_num= 7000, batch_size = 10, learning_rate = 0.5)
    #
    # linear_tensorflow(iter_num = 7000, batch_size = 25, learning_rate = 0.5)
    #
    # linear_tensorflow(iter_num = 7000, batch_size =25, learning_rate = 0.2)
    #
    #linear_tensorflow(iter_num = 7000, batch_size =25, learning_rate = 0.9)


 # ******************************first deep model ************************************************


    # first_deep_network(iter_num = 5000, batch_size = 10, learning_rate= 0.001)
    #
    #first_deep_network(iter_num = 10000, batch_size = 25, learning_rate= 0.001)
    #
    # first_deep_network(iter_num=5000, batch_size=50, learning_rate=0.001)
    #
    # first_deep_network(iter_num = 5000, batch_size = 25, learning_rate= 0.01)
    #
    # first_deep_network(iter_num = 5000, batch_size = 25, learning_rate= 0.001)
    #
    # first_deep_network(iter_num=5000, batch_size=25, learning_rate=0.0001)


# ******************************conv neural net ************************************************

    conv_neural_net(iter_num =10000,batch_size= 25, learning_rate= 0.001)


# ***********************************

    iter_num = 10000
    batch_size = [1 , 25, 50]
    learning_rate = 0.001
    train_loss = []
    test_loss = []
    accuracy_iter = []
    acc = []

    for batch in batch_size:

        train_loss_i, test_loss_i, accuracy_iter_i,acc_i = first_deep_network(iter_num, batch, learning_rate)
        train_loss.append(train_loss_i)
        test_loss.append(test_loss_i)
        accuracy_iter.append(accuracy_iter_i)
        acc.append(acc_i)


    print_linear_tensorflow_by_batch(3, train_loss, test_loss, accuracy_iter, iter_num, batch_size,
                                     learning_rate,acc)






