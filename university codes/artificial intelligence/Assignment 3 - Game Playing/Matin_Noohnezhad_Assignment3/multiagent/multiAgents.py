# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodValue = 0
        for i in range(newFood.width):
            for j in range(newFood.height):
                if (newFood[i][j]):
                    distance = abs(i - newPos[0]) + abs(j - newPos[1])
                    foodValue = foodValue + float(1 / float(distance))

        ghostValue = 0
        for i in range(len(newGhostStates)):
            distance = abs(newPos[0] - newGhostStates[i].configuration.getPosition()[0]) + abs(
                newPos[1] - newGhostStates[i].configuration.getPosition()[1])
            if (float(distance) != 0):
                ghostValue = ghostValue + float(5 / float(distance))

        return successorGameState.getScore() + foodValue - ghostValue


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
          gameState.isWin():
            Returns whether or not the game state is a winning state
          gameState.isLose():
            Returns whether or not the game state is a losing state
        """

        def maximizer(gameState1, depth):  # maximizer need to know game state and current depth.
            if (gameState1.isWin() or gameState1.isLose() or self.depth == depth):  # check if it is terminal state. In other words, if search has reached the depth limit which is stored in self.depth or pacman has won or pacman has lost
                return scoreEvaluationFunction(gameState1)  # return the score of this state. score of state is implemented in self.evaluationFunction()
            Moves = gameState1.getLegalActions(0)  # get legal moves of current state. remember that pacman is agent 0
            successors = [gameState1.generateSuccessor(0, action) for action in Moves]  # generate successor for each legal action of pacman
            scores1 = [minimizer(1, s, depth) for s in successors]  # maximizer feeds each of its successor states to minimizer. check inputs of minimizer
            return np.max(scores1)  # what does maximizer return ?

        def minimizer(gIndex, gameState1, depth):  # there are more than one ghosts. So minimizer takes the ghost index in addition to state and curent depth
            if (gameState1.isWin() or gameState1.isLose() or self.depth == depth):  # check if it is terminal state.
                return scoreEvaluationFunction(gameState1)  # just like maximizer

            Moves = gameState1.getLegalActions(gIndex)  # just like maximizer
            successors = [gameState1.generateSuccessor(gIndex, action) for action in Moves]

            # ** notice: pacman is agent 0. there are might be variuos number of agents in the game. agent 0 is considered to be our lovely pacman.
            # u can check number of agents with gameState.getNumAgents()
            # Consider there are 3 ghots. Their indexes would be 1 2 and 3
            # pacman is a maximizer and always takes max from minimizer.
            # however ghosts 1 2 take min from minimizer because after their turn is finished another ghost is plaing not pacman.
            # remeber ghosts are all in the same team and they wanna minimize score.
            # only the last ghost which is ghost 3 call maximizer, because after this ghost its pacman's turn to play....
            # remember the search tree of minimax for more than one ghost from the videos ....

            if (gIndex< gameState1.getNumAgents() -1):  # check the ghost index.
                scores1 = [minimizer(gIndex+1 , state , depth) for state in successors]  # who should be called on successors of ghost 1 and 2 action state? maximizer or minimizer ?
            else:
                scores1 = [maximizer(state,depth+1) for state in successors]  # who should be called on successors of ghost 3 action state? maximizer or minimizer ?
            return np.min(scores1)  # what does minimizer return ?

        legalMoves =  gameState.getLegalActions(0)# get legal moves of current state. remember that pacman is agent 0
        resultStates = [gameState.generateSuccessor(0, action) for action in legalMoves] # generate successor for each legal action of pacman

        scores = [minimizer(1, s , 0) for s in resultStates]  # who should be called on the successors of pacman? maximizer or minimizer
        best_action_index =  np.argmax(scores) # ------ check np.argmax function
        return legalMoves[best_action_index]# the index of best action from legalMoves list.

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"


        def maximizer(gameState1, depth,alpha,beta):
            if (gameState1.isWin() or gameState1.isLose() or depth == self.depth):# terminal state
                return scoreEvaluationFunction(gameState1)
            legalActions = gameState1.getLegalActions(0)
            currentScore = -float('inf') # set an initial score value
            for action in legalActions:
                resultState =gameState1.generateSuccessor(0, action)

                # iteratively find the maximum score of actions till now...
                a = minimizer(resultState,depth, 1 , alpha,beta)
                if a > currentScore: currentScore = a
                if (currentScore > beta): # hear u should prune
                    return currentScore
                if(currentScore>alpha): alpha = currentScore# update alpha
            return currentScore
        def minimizer(gameState1, depth, gIndex,alpha,beta):
            if (gameState1.isWin() or gameState1.isLose() or depth == self.depth):# terminal state
                return scoreEvaluationFunction(gameState1)
            legalMoves = gameState1.getLegalActions(gIndex)
            currentScore = float('inf') # set an initial score value

            for move in legalMoves:
                resultState = gameState1.generateSuccessor(gIndex, move)
                if (gIndex+1 == gameState1.getNumAgents()):
                    # iteratively find the minimum score of actions till now...
                    a = maximizer(resultState, depth+1, alpha, beta)
                    if(a<currentScore): currentScore = a
                else:
                    # iteratively find the minimum score of actions till now...
                    a = minimizer(resultState, depth, gIndex+1, alpha, beta)
                    if(a<currentScore):currentScore = a
                if (currentScore< alpha): # hear u should prune
                    return currentScore
                if(currentScore<beta):beta = currentScore #update beta
            return currentScore

        alpha = -float('inf') # initial alpha value
        beta = float('inf') # initial beta value
        legalActions = gameState.getLegalActions(0)

        score = -float('inf')# initial value
        bestAction = 0 # initial move

        for move in legalActions:
            resultState = gameState.generateSuccessor(0, move)
            b = score
            # iteratively find the maximum score of actions till now...
            temp = minimizer(resultState, 0, 1, alpha, beta)
            if(temp>score): score = temp
            # iteratively find the best action till now
            # if
            if(score > b):
                bestAction = move

            # if (alpha>score): # prune
            #     return -----
            alpha = score # update alpha

        return bestAction



        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maximizer(gameState1, depth):  # maximizer need to know game state and current depth.
            if (gameState1.isWin() or gameState1.isLose() or self.depth == depth):  # check if it is terminal state. In other words, if search has reached the depth limit which is stored in self.depth or pacman has won or pacman has lost
                return scoreEvaluationFunction(gameState1)  # return the score of this state. score of state is implemented in self.evaluationFunction()
            Moves = gameState1.getLegalActions(0)  # get legal moves of current state. remember that pacman is agent 0
            successors = [gameState1.generateSuccessor(0, action) for action in Moves]  # generate successor for each legal action of pacman
            scores1 = [expected(1, s, depth) for s in successors]  # maximizer feeds each of its successor states to minimizer. check inputs of minimizer
            return np.max(scores1)  # what does maximizer return ?

        def expected(gIndex, gameState1, depth):  # there are more than one ghosts. So minimizer takes the ghost index in addition to state and curent depth
            if (gameState1.isWin() or gameState1.isLose() or self.depth == depth):  # check if it is terminal state.
                return scoreEvaluationFunction(gameState1)  # just like maximizer

            Moves = gameState1.getLegalActions(gIndex)  # just like maximizer
            successors = [gameState1.generateSuccessor(gIndex, action) for action in Moves]

            # ** notice: pacman is agent 0. there are might be variuos number of agents in the game. agent 0 is considered to be our lovely pacman.
            # u can check number of agents with gameState.getNumAgents()
            # Consider there are 3 ghots. Their indexes would be 1 2 and 3
            # pacman is a maximizer and always takes max from minimizer.
            # however ghosts 1 2 take min from minimizer because after their turn is finished another ghost is plaing not pacman.
            # remeber ghosts are all in the same team and they wanna minimize score.
            # only the last ghost which is ghost 3 call maximizer, because after this ghost its pacman's turn to play....
            # remember the search tree of minimax for more than one ghost from the videos ....

            if (gIndex< gameState1.getNumAgents() -1):  # check the ghost index.
                scores1 = [expected(gIndex+1 , state , depth) for state in successors]  # who should be called on successors of ghost 1 and 2 action state? maximizer or minimizer ?
            else:
                scores1 = [maximizer(state,depth+1) for state in successors]  # who should be called on successors of ghost 3 action state? maximizer or minimizer ?
            return np.average(scores1)  # what does minimizer return ?

        legalMoves =  gameState.getLegalActions(0)# get legal moves of current state. remember that pacman is agent 0
        resultStates = [gameState.generateSuccessor(0, action) for action in legalMoves] # generate successor for each legal action of pacman

        scores = [expected(1, s , 0) for s in resultStates]  # who should be called on the successors of pacman? maximizer or minimizer
        best_action_index =  np.argmax(scores) # ------ check np.argmax function
        return legalMoves[best_action_index]# the index of best action from legalMoves list.

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # newFood = successorGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    foodValue = 0
    for i in range(currentGameState.getFood().width):
        for j in range(currentGameState.getFood().height):
            if (currentGameState.getFood()[i][j]):
                distance = abs(i - currentGameState.getPacmanPosition()[0]) + abs(j - currentGameState.getPacmanPosition()[1])
                foodValue = foodValue + float(1 / float(distance))

    ghostValue = 0
    for i in range(len(currentGameState.getGhostStates())):
        distance = abs(currentGameState.getPacmanPosition()[0] - currentGameState.getGhostStates()[i].configuration.getPosition()[0]) + abs(
            currentGameState.getPacmanPosition()[1] - currentGameState.getGhostStates()[i].configuration.getPosition()[1])
        if (float(distance) != 0):
            ghostValue = ghostValue + float(5 / float(distance))

    return currentGameState.getScore() + foodValue - ghostValue

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
