class History(object):
	"""
	Saves games History
	can be used to replay when learning
	"""

    def __init__(self, max_records):

        self._max_records = max_records
        self._size = 0
        self._index = 0

        self._history = np.recarray(self._max_records,
                                    dtype=[("prev_state", np.int32, STATE_DIM),
                                           ("new_state", np.int32, STATE_DIM),
                                           ("curr_action", np.int32, 1),
                                           ("reward", np.int32, 1),
                                           ("prev_legal", np.int32, ACTION_DIM),
                                           ("new_legal", np.int32, ACTION_DIM)])

        self._curr_moves = [None for i in range(self._max_records)]
        self._next_moves = [None for i in range(self._max_records)]


    def append(self, record_list):

        n = record_list.shape[0]

        # Make sure we have space for the records
        if n + self._index > self._max_records:
            free_space = self._max_records - self._index

            # Go over the records and add them
            self._history[self._index : self._index + free_space] = record_list[:free_space]
            self._index = 0

            # Go over the records and add them
            self._history[self._index : self._index + n - free_space] = record_list[free_space:]
            self._index = n - free_space

        else:
            # Go over the records and add them
            self._history[self._index : self._index + n] = record_list
            self._index += n

        self._size = min(self._max_records, self._size + n)


    def __len__(self):
        return self._size


    def get_batches(self, batch_size, batch_count):
        """
        Return list of random batches (; from the history.

        :param batch_size: The number of samples in each biatch
        :param batch_count: The number of bitches of size batch_size to return
        :return: a generator of batch_count bitches each of size batch_size
        """

        # limit batch size to the amount of records
        if batch_size > self._size:
            batch_size = self._size

        p = np.random.permutation(self._size)
        history = self._history[p]

        perm_index = 0

        # Generate as much batches as we asked
        for i in range(batch_count):

            # Generate a new permutation if we depleted the current
            if perm_index + batch_size > self._size:
                p = np.random.permutation(self._size)
                history = self._history[p]
                perm_index = 0

            yield history[perm_index : perm_index + batch_size]

            perm_index += batch_size

