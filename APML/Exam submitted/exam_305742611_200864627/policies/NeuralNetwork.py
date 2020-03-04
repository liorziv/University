class NeuralNetwork(object):

	'''
	The nerounal network the agaent uses in order to predict the next move
	'''
    def __init__(self, network_params):

        self._session = tf.Session()

        # Random weights if no parameters were passed
        if not network_params:
            network_params = {}

            network_params["W1"] = self.weight_variable([STATE_DIM, HIDDEN_LAYER_SIZE])
            network_params["B1"] = self.bias_variable([HIDDEN_LAYER_SIZE])

            network_params["W2"] = self.weight_variable([HIDDEN_LAYER_SIZE, HIDDEN_LAYER_2_SIZE])
            network_params["B2"] = self.bias_variable([HIDDEN_LAYER_2_SIZE])

            network_params["W3"] = self.weight_variable([HIDDEN_LAYER_2_SIZE, ACTION_DIM])
            network_params["B3"] = self.bias_variable([ACTION_DIM])

        else:
            for key, val in network_params.items():
                network_params[key] = tf.Variable(tf.constant(val))

        self.input = tf.placeholder(tf.float32, shape=(None, STATE_DIM))

        # self.input_1 = tf.placeholder(tf.float32, shape=(None, STATE_DIM // 2))
        # self.input_2 = tf.placeholder(tf.float32, shape=(None, STATE_DIM // 2))

        # Hidden layer 1
        self.W_1 = network_params["W1"]
        self.B_1 = network_params["B1"]


        layer_1 = tf.nn.relu(tf.matmul(self.input, self.W_1) + self.B_1)

        # Hidden layer 2
        self.W_2 = network_params["W2"]
        self.B_2 = network_params["B2"]

        layer_2 = tf.nn.relu(tf.matmul(layer_1, self.W_2) + self.B_2)

        # Hidden layer 3
        self.W_3 = network_params["W3"]
        self.B_3 = network_params["B3"]

        self.Q = tf.matmul(layer_2, self.W_3) + self.B_3

        # Limit moves to the legal ones
        self.legal_moves = tf.placeholder(tf.float32, shape=(None, ACTION_DIM))

        self.legal_Q = self.legal_moves + self.Q
        self._Q_max = tf.reduce_max(self.legal_Q, axis=1)
        self._Q_argmax = tf.argmax(self.legal_Q, axis=1)


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=1.0)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial)


    def init_variables(self):
        self._session.run(tf.global_variables_initializer())


    def train(self, optimizer, feed_dict):
        self._session.run(optimizer, feed_dict=feed_dict)


    def predict_q_max(self, state, legal_moves):
        return self._session.run(self._Q_max, feed_dict={self.input: state,
                                                         self.legal_moves: legal_moves})


    def predict(self, state, legal_moves, eps_greedy=0):

        if np.random.random() < eps_greedy:
            return [np.random.choice(np.where(legal_moves[0, :] == 0)[0])]

        move, q = self._session.run([self._Q_argmax, self.Q], feed_dict={self.input: state,
                                                                         self.legal_moves: legal_moves})

        return move

    def serialize(self):
        network_params = {}

        network_params["W1"], \
        network_params["W2"], \
        network_params["W3"], \
        network_params["B1"], \
        network_params["B2"], \
        network_params["B3"] = self._session.run([self.W_1, self.W_2, self.W_3,
                                                  self.B_1, self.B_2, self.B_3])

        return network_params
