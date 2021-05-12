__author__ = 'L.Z & A.B'

import tensorflow as tf
import numpy as np

WIN_MASK = np.ones(4)
STATE_DIM = 42
ROWS = 6
ACTION_DIM = 7
HIDDEN_LAYER_SIZE = 128
EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2


class DeepQLearner(object):

    def __init__(self,
                 gamma,
                 learning_rate,
                 eps_greedy,
                 history_size,
                 batch_size,
                 batch_count,
                 network_params):

        self._gamma = gamma
        self._batch_size = batch_size
        self._batch_count = batch_count
        self.eps_greedy = eps_greedy
        self.learning_rate = learning_rate

        self._network = NeuralNetwork(network_params)
        self._history = History(max_records=history_size)

        self._tf_action = tf.placeholder(tf.int32, shape=(None,))
        self._tf_Q_max = tf.placeholder(tf.float32, shape=(None,))

        self._tf_Q_act = tf.boolean_mask(self._network.Q, tf.one_hot(indices=self._tf_action,
                                                                     depth=ACTION_DIM,
                                                                     dtype=tf.bool,
                                                                     on_value=True,
                                                                     off_value=False,
                                                                     axis=-1))


        self._loss = tf.reduce_mean((self._tf_Q_act - self._tf_Q_max) ** 2)

        # TODO: drop learning rate later
        self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._train_step = self._optimizer.minimize(self._loss)


        self._network.init_variables()


    def iteration(self):

        for batch, curr_moves, next_moves in self._history.get_batches(batch_size=self._batch_size,
                                                                       batch_count=self._batch_count):

            q_max = batch.reward + self._gamma * self._network.predict(batch.next_state, np.array(next_moves))

            feed_dict = {self._network.input: batch.curr_state,
                         self._network.legal_moves: curr_moves,
                         self._tf_action: batch.curr_action,
                         self._tf_Q_max: q_max}

            self._network.train(optimizer=self._train_step,
                                feed_dict=feed_dict)

        # TODO: decay eps-greedy
        self.eps_greedy *= 0.9


    def append_history(self, curr_state, next_state, curr_action, reward, curr_move, next_move):
        self._history.append(np.array([(curr_state,
                                        next_state,
                                        curr_action,
                                        reward)],
                             dtype=[("curr_state", np.float32, 2*ACTION_DIM),
                                    ("next_state", np.float32, 2*ACTION_DIM),
                                    ("curr_action", np.int32, 1),
                                    ("reward", np.float32, 1)]),
                             [curr_move],
                             [next_move])


    def single_move(self, state, legal_moves):

        move = self._network.predict(state=state,
                                     legal_moves=legal_moves,
                                     eps_greedy=self.eps_greedy)

        return move[0]


    def serialize(self):
        return [self.eps_greedy, self.learning_rate, self._network.serialize()]




class NeuralNetwork(object):

    def __init__(self, network_params):

        self._session = tf.Session()

        # Random weights if no parameters were passed
        if not network_params:
            network_params = {}
            network_params["W1"] = tf.random_normal([2*ACTION_DIM, HIDDEN_LAYER_SIZE])
            network_params["B1"] = tf.random_normal([HIDDEN_LAYER_SIZE])

            network_params["W2"] = tf.random_normal([HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE])
            network_params["B2"] = tf.random_normal([HIDDEN_LAYER_SIZE])

            network_params["W3"] = tf.random_normal([HIDDEN_LAYER_SIZE, ACTION_DIM])
            network_params["B3"] = tf.random_normal([ACTION_DIM])

        else:
            for key, val in network_params.items():
                network_params[key] = tf.constant(val)

        self.input = tf.placeholder(tf.float32, shape=(None, 2*ACTION_DIM))

        # Hidden layer 1
        self.W_1 = tf.Variable(network_params["W1"])
        self.B_1 = tf.Variable(network_params["B1"])

        layer_1 = tf.nn.relu(tf.matmul(self.input, self.W_1) + self.B_1)

        # Hidden layer 2
        self.W_2 = tf.Variable(network_params["W2"])
        self.B_2 = tf.Variable(network_params["B2"])

        layer_2 = tf.nn.relu(tf.matmul(layer_1, self.W_2) + self.B_2)

        # Readout layer
        self.W_3 = tf.Variable(network_params["W3"])
        self.B_3 = tf.Variable(network_params["B3"])

        self.Q = tf.matmul(layer_2, self.W_3) + self.B_3


        # Limit moves to the legal ones
        self.legal_moves = tf.placeholder(tf.float32, shape=(None, ACTION_DIM))

        self.legal_Q = self.legal_moves + self.Q
        self._Q_max = tf.reduce_max(self.legal_Q, axis=1)
        self._Q_argmax = tf.argmax(self.legal_Q, axis=1)


    def init_variables(self):
        self._session.run(tf.global_variables_initializer())


    def train(self, optimizer, feed_dict):
        self._session.run(optimizer, feed_dict=feed_dict)


    def predict_q_max(self, state, legal_moves):
        return self._session.run(self._Q_max, feed_dict={self.input: state, self.legal_moves: legal_moves})


    def predict(self, state, legal_moves, eps_greedy=0):

        if np.random.random() < eps_greedy:
            return [np.random.choice(np.where(legal_moves == 0)[0])]

        move, q = self._session.run([self._Q_argmax, self.Q], feed_dict={self.input: state, self.legal_moves: legal_moves})
        # print("q", q, "max", move)
        return move

    def serialize(self):
        network_params = {}

        network_params["W1"], \
        network_params["W2"], \
        network_params["W3"], \
        network_params["B1"], \
        network_params["B2"], \
        network_params["B3"] = self._session.run([self.W_1, self.W_2, self.W_3, self.B_1, self.B_2, self.B_3])

        return network_params


class History(object):

    def __init__(self, max_records):

        self._max_records = max_records
        self._size = 0

        self._history = np.recarray(self._max_records,
                                    dtype=[("curr_state", np.int32, 2*ACTION_DIM),
                                           ("next_state", np.int32, 2*ACTION_DIM),
                                           ("curr_action", np.int32, 1),
                                           ("reward", np.int32, 1)])

        self._curr_moves = []
        self._next_moves = []


    def clear(self):
        self._size = 0
        self._curr_moves = []
        self._next_moves = []


    def append(self, record_list, curr_moves, next_moves):

        n = record_list.shape[0]

        # Make sure we have space for the records
        if n + self._size > self._max_records:
            self.clear()

        # Go over the records and add them
        self._history[self._size : self._size + n] = record_list
        self._curr_moves += curr_moves
        self._next_moves+= next_moves
        self._size += n


    def __len__(self):
        return self._size


    def get_batches(self, batch_size, batch_count):
        """
        Return list of random bitches (; from the history.

        :param batch_size: The number of samples in each biatch
        :param batch_count: The number of bitches of size batch_size to return
        :return: a generator of batch_count bitches each of size batch_size
        """

        # limit batch size to the amount of records
        if batch_size > self._size:
            batch_size = self._size

        p = np.random.permutation(self._size)
        history = self._history[p]
        curr_moves = [self._curr_moves[i] for i in p]
        next_moves = [self._next_moves[i] for i in p]

        perm_index = 0

        # Generate as much batches as we asked
        for i in range(batch_count):

            # Generate a new permutation if we depleted the current
            if perm_index + batch_size > self._size:
                p = np.random.permutation(self._size)
                history = self._history[p]
                curr_moves = [self._curr_moves[i] for i in p]
                next_moves = [self._next_moves[i] for i in p]
                perm_index = 0

            yield history[perm_index : perm_index + batch_size], \
                  curr_moves[perm_index : perm_index + batch_size], \
                  next_moves[perm_index : perm_index + batch_size]

            perm_index += batch_size


if __name__ == "__main__":
    hist = History(max_records=100)

    records = [([0] * STATE_DIM, [1] * STATE_DIM, 1, 0),
               ([1] * STATE_DIM, [2] * STATE_DIM, 2, 0),
               ([2] * STATE_DIM, [3] * STATE_DIM, 3, 0),
               ([3] * STATE_DIM, [4] * STATE_DIM, 4, 0),
               ([4] * STATE_DIM, [5] * STATE_DIM, 5, 0),
               ([5] * STATE_DIM, [6] * STATE_DIM, 6, 0),
               ([6] * STATE_DIM, [7] * STATE_DIM, 7, 0),
               ([7] * STATE_DIM, [8] * STATE_DIM, 8, 0),
               ([8] * STATE_DIM, [9] * STATE_DIM, 9, 0)]

    curr_moves = [[1],
                  [2],
                  [3],
                  [4],
                  [5],
                  [6],
                  [7],
                  [8],
                  [9]]

    next_moves = [[1],
                  [2],
                  [3],
                  [4],
                  [5],
                  [6],
                  [7],
                  [8],
                  [9]]

    records = np.array(records, dtype=[("curr_state", np.float32, STATE_DIM),
                                       ("next_state", np.float32, STATE_DIM),
                                       ("curr_action", np.int32, 1),
                                       ("reward", np.float32, 1)])

    hist.append(records, curr_moves, next_moves)

    for batch, curr_moves, next_moves in hist.get_batches(2, 5):
        print("*" * 50)
        print(batch.curr_action)
        print(curr_moves)
        print(next_moves)



