class DeepQLearner(object):
	'''
	The deep learning agent which predicts the next move
	using NeuralNetwork class
	'''

    def __init__(self,
                 gamma,
                 learning_rate,
                 eps_greedy,
                 history_size,
                 batch_size,
                 batch_count,
                 network_params):

        self._gamma = gamma
        self.batch_size = batch_size
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

        for batch in self._history.get_batches(batch_size=self.batch_size,
                                                                       batch_count=self._batch_count):

            q_max = batch.reward + self._gamma * self._network.predict(batch.new_state, batch.new_legal)

            feed_dict = {self._network.input: batch.prev_state,
                         self._network.legal_moves: batch.prev_legal,
                         self._tf_action: batch.curr_action,
                         self._tf_Q_max: q_max}

            self._network.train(optimizer=self._train_step,
                                feed_dict=feed_dict)

        # Decay up to random once every 5 turns
        self.eps_greedy = min(self.eps_greedy * 0.9999, 0.05)


    def append_history(self, prev_state, new_state, curr_action, reward, curr_move, next_move):

        self._history.append(np.array([(prev_state.flatten(),
                                        new_state.flatten(),
                                        curr_action,
                                        reward,
                                        curr_move.flatten(),
                                        next_move.flatten())],
                             dtype=[("prev_state", np.float32, STATE_DIM),
                                    ("new_state", np.float32, STATE_DIM),
                                    ("curr_action", np.int32, 1),
                                    ("reward", np.float32, 1),
                                    ("prev_legal", np.float32, ACTION_DIM),
                                    ("new_legal", np.float32, ACTION_DIM)]))


    def single_move(self, state, legal_moves):

        move = self._network.predict(state = state,
                                     legal_moves=legal_moves,
                                     eps_greedy=self.eps_greedy)

        return move[0]


    def serialize(self):
        return [self.eps_greedy, self.learning_rate, self._network.serialize()]
