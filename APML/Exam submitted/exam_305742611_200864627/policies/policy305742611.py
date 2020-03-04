from policies import base_policy as bp
import numpy as np
import tensorflow as tf
import pickle



EMPTY_VAL = 0
WIN_MASK = np.ones(4)
ROWS = 6
ACTION_DIM = 7
STATE_DIM = 14
HIDDEN_LAYER_SIZE = 128
HIDDEN_LAYER_2_SIZE = 64


class policy305742611(bp.Policy):
    """
    A test agent that will give us a good grade (hopefully =] )
    """

    def cast_string_args(self, policy_args):

        policy_args['save_to'] = str(policy_args['save_to']) if 'save_to' in policy_args else 'model_305742611_200864627.pkl'
        policy_args['load_from'] = str(policy_args['load_from']) if 'load_from' in policy_args else 'models/model_305742611_200864627.pkl'

        return policy_args


    def init_run(self):

        rival = {1: 2, 2: 1}
        self.rival = rival[self.id]


        try:
            self.log("trying to load %s" % self.load_from, "STATUS")
            eps_greedy, learning_rate, network_params = pickle.load(open(self.load_from, 'rb'))

        except:
            self.log("No model loaded, initializing new learner instance...", "STATUS")
            eps_greedy, learning_rate, network_params = 1, 0.0001, None


        if self.mode == "test":
            eps_greedy = 0

        self._learner = DeepQLearner(gamma=0.999,
                                     learning_rate=learning_rate,
                                     eps_greedy=eps_greedy,
                                     history_size=100000,
                                     batch_size=200,
                                     batch_count=10,
                                     network_params=network_params)

        self.log("Init Finished", 'STATUS')


    def get_legal_mask(self, board_state):

        mask = (board_state== bp.EMPTY_VAL).any(axis=0)
        result = np.full(ACTION_DIM, -np.infty)
        result[mask] = 0

        return result


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):


        if too_slow:
            self._learner.batch_size -= 1


        if prev_state is not None:
            curr_move = np.array(self.get_legal_mask(prev_state))
            next_move = np.array(self.get_legal_mask(new_state))

            prev_state = self.process_state(prev_state)
            new_state = self.process_state(new_state)

            self._learner.append_history(prev_state, new_state, prev_action, reward, curr_move, next_move)


        self._learner.iteration()


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:
            next_move = np.array(self.get_legal_mask(new_state))
            new_state = self.process_state(new_state)

            if self.mode == "train":
                if prev_state is not None:
                    curr_move = np.array(self.get_legal_mask(prev_state))
                    prev_state = self.process_state(prev_state)

                    self._learner.append_history(prev_state, new_state, prev_action, reward, curr_move, next_move)

            return self._learner.single_move(state=new_state,
                                             legal_moves=next_move.reshape(1, ACTION_DIM))

        # TODO: remove this
        except NotImplementedError:
            self.log("Oh No! Exception!", 'EXCEPTION')


    def process_state(self, state):

        winning_moves = self.get_winning_moves(state).reshape((1, 14))

        return winning_moves


    def save_model(self):

        self.log("Saving model", 'Status')

        return self._learner.serialize(), self.save_to

    def proximity_factor(self, state):

        proximity = np.zeros((2, ACTION_DIM))

        # find legal actions:
        legal_actions = self.get_legal_moves(state)

        # in case of weird problems and draws (no legal actions):
        if len(legal_actions) == 0:
            return np.zeros((2, ACTION_DIM))


        for action in legal_actions:
            for player_id in [self.id, self.rival]:

                row = np.max(np.where(state[:, action] == EMPTY_VAL))
                proximity[player_id - 1, action] = (state[max(row - 1, 0):row + 2, max(action - 1, 0): action + 2] == player_id).sum()

        return proximity

    def distance_from_top(self, state):

        distance = np.zeros((1, ACTION_DIM))

        for action in self.get_legal_moves(state):
            distance[0, action] = np.max(np.where(state[:, action] == EMPTY_VAL)) + 1

        return distance

    def check_for_win(self,board, player_id, action):
        """
        check the board to see if last move was a winning move.
        :param board: the new board
        :param player_id: the player who made the move
        :param col: his action
        :return: True iff the player won with his move
        """

        row = 0

        # check which row was inserted last:
        for i in range(ROWS):
            if board[ROWS - 1 - i, action] == EMPTY_VAL:
                row = ROWS - i
                break

        # check horizontal:
        vec = board[row, :] == player_id
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return True

        # check vertical:
        vec = board[:, action] == player_id
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return True

        # check diagonals:
        vec = np.diagonal(board, action - row) == player_id
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return True
        vec = np.diagonal(np.fliplr(board), ACTION_DIM - action - 1 - row) == player_id
        if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
            return True

        return False

    def make_move(self,board, action, player_id):
        """
        return a new board with after performing the given move.
        :param board: original board
        :param action: move to make (column)
        :param player_id: player that made the move
        :return: new board after move was made
        """
        row = np.max(np.where(board[:, action] == EMPTY_VAL))
        new_board = np.copy(board)
        new_board[row, action] = player_id

        return new_board

    def get_winning_moves(self,board):

        # find legal actions:
        legal_actions = self.get_legal_moves(board)

        # in case of weird problems and draws (no legal actions):
        if len(legal_actions) == 0:
            return np.zeros((2, ACTION_DIM))


        win_moves = np.zeros((2, ACTION_DIM))
        for action in legal_actions:
            for player_id in [self.id, self.rival]:
                new_board = self.make_move(board, action, player_id)
                if self.check_for_win(new_board, player_id, action):
                    win_moves[player_id -1][action] = 1

        return win_moves

    def get_series(self, state):

        # find legal actions:
        legal_actions = self.get_legal_moves(state)

        # in case of weird problems and draws (no legal actions):
        if len(legal_actions) == 0:
            return 0, 0

        series_moves = np.zeros((2, ACTION_DIM))

        for action in legal_actions:
            for player_id in [self.id, self.rival]:
                new_state = self.make_move(state, action, player_id)

                for mask_size in range(3, 5):
                    if self.get_max_series(new_state, player_id, action, np.ones(mask_size)):
                        series_moves[player_id - 1][action] = mask_size

        return series_moves

    def get_max_series(self,board, player_id, action, mask):
        """
        check the board to see if last move was a winning move.
        :param board: the new board
        :param player_id: the player who made the move
        :param col: his action
        :return: True iff the player won with his move
        """

        n = mask.shape[0]
        row = 0

        # check which row was inserted last:
        for i in range(ROWS):
            if board[ROWS - 1 - i, action] == EMPTY_VAL:
                row = ROWS - i
                break

        # check horizontal:
        vec = board[row, :] == player_id
        if np.any(np.convolve(mask, vec, mode="valid") == n):
            return True

        # check vertical:
        vec = board[:, action] == player_id
        if np.any(np.convolve(mask, vec, mode="valid") == n):
            return True

        # check diagonals:
        vec = np.diagonal(board, action - row) == player_id
        if np.any(np.convolve(mask, vec, mode="valid") == n):
            return True

        vec = np.diagonal(np.fliplr(board), ACTION_DIM - action - 1 - row) == player_id
        if np.any(np.convolve(mask, vec, mode="valid") == n):
            return True

        return False




