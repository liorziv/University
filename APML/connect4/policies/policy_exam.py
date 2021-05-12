__author__ = 'amirbar'

from policies import base_policy as bp
import numpy as np
import pickle

from policies.DeepQLearner import DeepQLearner, ACTION_DIM

WIN_MASK = np.ones(4)
STATE_DIM = 42
ROWS = 6
ACTION_DIM = 7
EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2

class TestAgent(bp.Policy):
    def check_for_win(self, board, player_id, action):
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
                row = ROWS - i - 1
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
    def get_legal_actions(self, board):
        legal_actions = np.array(np.where(board[0, :] == EMPTY_VAL))
        legal_actions = np.reshape(legal_actions, (legal_actions.size,))
        # in case of weird problems and draws (no legal actions):

        return legal_actions

    def get_winning_moves(self,board):

        # find legal actions:
        legal_actions_1 = self.get_legal_actions(board)
        if len(legal_actions_1) == 0:
            return 0, 0

        win_moves = np.zeros((2,ACTION_DIM))
        for player_id_1 in [PLAYER1_ID, PLAYER2_ID]:

            for action1 in legal_actions_1:

                new_board = self.make_move(board, action1, player_id_1)
                if (self.check_for_win(new_board, player_id_1, action1)):
                    win_moves[player_id_1 - 1][action1] = 1
                    break
                legal_actions_2 =self.get_legal_actions(new_board)
                if len(legal_actions_2) == 0:
                    break
                #print("legal actions 2" + str(legal_actions_2))

                player_id_2  = 3 - player_id_1
                for action2 in legal_actions_2:
                    #print("player " + str(player_id) + " action2 : " + str(action2)+" action1 : "+str(action1))
                    new_board_2 = self.make_move(new_board, action2, player_id_2)
                    if(action2 in self.get_legal_actions(new_board_2)):
                        new_board_3 = self.make_move(new_board_2, action2, player_id_1)
                        if (self.check_for_win(new_board_3, player_id_1, action2)):
                            win_moves[player_id_1 - 1][action2] = 7
                            print('s#'*10)
                            print(board)
                            print('#' * 10)
                            print(new_board)
                            print('#' * 10)
                            print(new_board_2)
                            print('e#' * 10)
                            print(new_board_3)
                            # print("player " + str(player_id_1) + " action1 : " + str(action1))
                            # print("player " + str(player_id_2) + " action2 : " + str(action2))
                            # print("player " + str(player_id_1) + " action3 : " + str(action2))
                            # print(new_board_3)

        return win_moves.reshape(1,ACTION_DIM*2)


    """
    A test agent that will give us a good grade (hopefully =] )
    """

    def cast_string_args(self, policy_args):

        policy_args['save_to'] = str(policy_args['save_to']) if 'save_to' in policy_args else 'test.model.pkl'
        # policy_args['load_from'] = str(policy_args['load_from']) if 'load_from' in policy_args else 'models/test.model.pkl'

        return policy_args


    def init_run(self):

        try:
            self.log("trying to load %s" % self.load_from, "STATUS")
            eps_greedy, learning_rate, network_params = pickle.load(open(self.load_from, 'rb'))

        except BaseException:
            eps_greedy, learning_rate, network_params = 0.01, 0.005, None
            self.log("No model loaded, initializing new learner instance...", "STATUS")


        self._learner = DeepQLearner(gamma=0.999,
                                     learning_rate=learning_rate,
                                     eps_greedy=eps_greedy,
                                     history_size=10000,
                                     batch_size=20,
                                     batch_count=10,
                                     network_params=network_params)

        self.log("Init Finished", 'STATUS')


    def get_legal_mask(self, board_state):

        mask = (board_state== bp.EMPTY_VAL).any(axis=0)
        result = np.full(ACTION_DIM, -np.infty)
        result[mask] = 0

        return result


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        #self.log("------> In learn", 'STATUS')

        if prev_state is not None:
                curr_move = np.array(self.get_legal_mask(prev_state))
                next_move = np.array(self.get_legal_mask(new_state))
                prev_winning_moves = self.get_winning_moves(prev_state)
                new_winning_moves = self.get_winning_moves(new_state)

                self._learner.append_history(prev_winning_moves, new_winning_moves, prev_action, reward, curr_move, next_move)

        self._learner.iteration()

        #self.log("------> Finished learn", 'STATUS')


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:

            next_move = np.array(self.get_legal_mask(new_state))
            new_winning_moves = self.get_winning_moves(board=new_state)

            if prev_state is not None:
                curr_move = np.array(self.get_legal_mask(prev_state))
                prev_winning_moves =  self.get_winning_moves(board=prev_state)

                self._learner.append_history(prev_winning_moves, new_winning_moves, prev_action, reward, curr_move, next_move)
            #winning_moves = self.get_winning_moves(board=new_state)
            #if(np.sum(new_winning_moves) != 0):
                # print('*' * 50 + 'moves')
                # print(new_winning_moves.reshape(2,ACTION_DIM))
                # print('*'*50 + 'board')
                # print(new_state)
            move = self._learner.single_move(new_winning_moves, legal_moves=next_move.reshape(1, ACTION_DIM))

            if move not in self.get_legal_moves(new_state):
                print(new_state)
                print(move)
                print(next_move.reshape(1, ACTION_DIM))
                return None

            return move

        except NotImplementedError:
            self.log("Oh No! Exception!", 'EXCEPTION')


    def save_model(self):

        self.log("Saving model", 'Status')

        return self._learner.serialize(), self.save_to




if __name__ == "__main__":

    ta = TestAgent(policy_args={},
                   stateq=None,
                   actq=None,
                   modelq=None,
                   logq=None,
                   id=None,
                   mode=None,
                   game_duration=None)
    ta.init_run()

    print(ta.act(round=1,
           prev_state = np.array([0.] * 42).reshape(1, 42),
           prev_action = 1,
           reward=0,
           new_state=np.array([1.] * 42).reshape(1, 42),
           too_slow=False))


