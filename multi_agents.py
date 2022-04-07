import math

import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    
    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """
        
        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()
        
        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best
        
        "Add more of your code here if you want to"
        
        return legal_moves[chosen_index]
    
    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        
        # Useful information you can extract from a GameState (game_state.py)
        
        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        board_size = len(board) * len(board[0])
        # count_twos = 0
        # for r in board:
        #     for cell in r:
        #         if cell == 2:
        #             count_twos += 1
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score
        free_tiles = successor_game_state.get_empty_tiles()
        free_tiles = len(free_tiles[0])
        taken_tiles = board_size - free_tiles
        return (score / taken_tiles) ** 2 * (max_tile + free_tiles) ** 2


def score_evaluation_function(current_game_state):
    """
    we don't change this function
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """
    
    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth
    
    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        action = self.helper_minmax(game_state, 1)[0]
        if action is None:
            return Action.STOP
        return action
    
    def helper_minmax(self, game_state, current_depth):
        possible_actions = game_state.get_legal_actions(0)  # possible actions for our agent
        max_score_for_agent = -math.inf
        best_move_for_agent = None
        for action in possible_actions:
            child_state = game_state.generate_successor(0, action)
            min_choice_for_opp = math.inf
            for opponent_action in child_state.get_legal_actions(1):
                grandchild = child_state.generate_successor(1, opponent_action)  # the states the opponent should
                # choose between
                if current_depth == self.depth:
                    grandchild_score = self.evaluation_function(grandchild)
                else:
                    grandchild_score = self.helper_minmax(grandchild, current_depth + 1)[1]
                min_choice_for_opp = min(min_choice_for_opp, grandchild_score)
            
            if max_score_for_agent < min_choice_for_opp:
                max_score_for_agent = min_choice_for_opp
                best_move_for_agent = action
        return best_move_for_agent, max_score_for_agent


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        score, action = self.alpha_beta_helper(game_state, self.depth, alpha=-math.inf, beta=math.inf, player_index=0)
        if action is None:  # if we see that we will get stuck in self.depth moves
            return Action.STOP
        return action
    
    def alpha_beta_helper(self, game_state, current_depth, alpha, beta, player_index):
        possible_actions = game_state.get_legal_actions(player_index) # possible actions for current player
        if current_depth == 0 or not possible_actions:
            return self.evaluation_function(game_state), None
        if player_index == 0:  # our agent - max player
            best_move_for_agent = None
            for action in possible_actions:  # for each child of node
                child_state = game_state.generate_successor(0, action)  # state after our agent's action
                child_score, child_move = self.alpha_beta_helper(child_state, current_depth - 1, alpha, beta,
                                                                 player_index=1)
                if child_score > alpha:
                    alpha = child_score
                    best_move_for_agent = action
                if beta <= alpha:
                    break
            return alpha, best_move_for_agent
        if player_index == 1:  # the opponent - min player
            min_move_for_opp = None
            for action in possible_actions:
                child_for_opp = game_state.generate_successor(1, action)
                child_score, child_move = self.alpha_beta_helper(child_for_opp, current_depth, alpha, beta,
                                                                 player_index=0)
                if child_score < beta:
                    beta = child_score
                    min_move_for_opp = action
                if beta <= alpha:
                    break
            return beta, min_move_for_opp
            
            
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """
    
    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action = self.helper_expect(game_state, 1)[0]
        if action is None:
            print("stop the game")
            return Action.STOP
        return action

    def helper_expect(self, game_state, current_depth):
        possible_actions = game_state.get_legal_actions(0)  # possible actions for our agent
        max_score_for_agent = -math.inf
        best_move_for_agent = None
        for action in possible_actions:
            child_state = game_state.generate_successor(0, action)
            mean_score_after_opponents_turn = 0
            opponent_branches = child_state.get_legal_actions(1)
            for opponent_action in opponent_branches:
                grandchild = child_state.generate_successor(1, opponent_action)  # the states the opponent should
                # choose between
                grandchild_score = self.evaluation_function(grandchild)
                if current_depth != self.depth:
                    grandchild_score = max(self.helper_expect(grandchild, current_depth + 1)[1], grandchild_score)
                mean_score_after_opponents_turn += grandchild_score
            if len(opponent_branches) != 0:
                mean_score_after_opponents_turn /= len(opponent_branches)
        
            if max_score_for_agent < mean_score_after_opponents_turn:
                max_score_for_agent = mean_score_after_opponents_turn
                best_move_for_agent = action
        return best_move_for_agent, max_score_for_agent


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    board = current_game_state.board
    board_size = len(board) * len(board[0])
    # count_twos = 0
    # for r in board:
    #     for cell in r:
    #         if cell == 2:
    #             count_twos += 1
    max_tile = current_game_state.max_tile
    score = current_game_state.score
    free_tiles = current_game_state.get_empty_tiles()
    free_tiles = len(free_tiles[0])
    taken_tiles = board_size - free_tiles
    return (score / taken_tiles) ** 2 * (max_tile + free_tiles) ** 2


# Abbreviation
better = better_evaluation_function
