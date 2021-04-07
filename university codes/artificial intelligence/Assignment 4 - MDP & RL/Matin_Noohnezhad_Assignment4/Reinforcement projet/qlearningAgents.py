# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.values = util.Counter() # initialize all values[(state,action)] with 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[(state,action)]

        # return proper value.
        # the current values of each (state,action) are stored in self.values[]
        # util.Counter() initializes every (state,action) with 0. So unseen (state,actions) have 0 Qvalue by default

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        actions = self.getLegalActions(state)
        if len(actions) == 0:# if action is empty ( terminal state)
            return 0
        # otherwise get Q(state,action) for all possible actions and return the maximum Q-value as the value of this state
        q_values = (self.getQValue(state,action) for action in actions)
        max_q_value = -float('inf')

        for a in q_values:
            if a>max_q_value:
                max_q_value = a
        return max_q_value


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        actions = self.getLegalActions(state)
        if  len(actions) == 0 :# no legal action from this state
            return None

        # select actions with maximum Qvalue and if several actions give the same Qvalue retrun one of them randomly
        # for example if Qvalue of (state,up): 2, (state,down):5 , (state,left):5 (state,right):-2 you should randomly choose between down and left


        q_values = (self.getQValue(state, action) for action in actions)
        # max_q_values_indices = np.argmax(q_values)

        max = -float('inf')
        for q in q_values:
            if(q>max):
                max = q

        action_list = []
        for action in actions:
            if(self.getQValue(state,action) == max):
                action_list.append(action)

        if(len(action_list) == 1):
            return action_list[0]
        else:
            return random.choice(action_list)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        "*** YOUR CODE HERE ***"
        # Pick Action
        actions = self.getLegalActions(state)
        if len(actions) == 0:  # no legal action from this state
            return None

        if(util.flipCoin(self.epsilon)):
            a = random.choice(actions)
            return a
        else:
            return self.computeActionFromQValues(state)


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        self.values[(state,action)] = ((1- self.alpha)*self.values[(state,action)]) + (self.alpha * (reward + self.discount* self.computeValueFromQValues(nextState)))




        # you might wanna check slide 8 of Lecture Notes:
        #http://ai.berkeley.edu/slides/Lecture%2011%20--%20Reinforcement%20Learning%20II/SP14%20CS188%20Lecture%2011%20--%20Reinforcement%20Learning%20II.pptx

        #update self.values

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        sum = 0
        for key in features:
            sum = sum + (features[key] * self.getWeights()[key])

        return sum



        # return self.getWeights()[(state, action)] * features[(state,action)]


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        if (nextState == 'TERMINAL_STATE'):
            new_value = reward
            old_value = self.getQValue(state, action)
            difference = new_value - old_value
            features = self.featExtractor.getFeatures(state, action)
            for key in features:
                self.getWeights()[key] = self.getWeights()[key] + (
                        self.alpha * difference * features[key])

        else:
            new_value = reward + (self.discount * self.computeValueFromQValues(nextState))
            old_value = self.getQValue(state,action)
            difference = new_value - old_value

            features = self.featExtractor.getFeatures(state, action)
            for key in features:
                self.getWeights()[key] = self.getWeights()[key] + (self.alpha*difference*features[key])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
