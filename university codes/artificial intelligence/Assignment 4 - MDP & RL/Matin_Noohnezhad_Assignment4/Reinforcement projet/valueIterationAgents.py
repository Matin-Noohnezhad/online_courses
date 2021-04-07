# valueIterationAgents.py
# -----------------------
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


import mdp, util
import numpy as np

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # in each iteration you should update values of each state.
        # Important: Use the "batch" version of value iteration where each vector Vk is computed from a fixed vector Vk-1 (like in lecture),
        # not the "online" version where one single weight vector is updated in place.
        # This means that when a state's value is updated in iteration k based on the values of its successor states,
        # the successor state values used in the value update computation should be those from iteration k-1
        # (even if some of the successor states had already been updated in iteration k).
        # The difference is discussed in Sutton & Barto in the 6th paragraph of chapter 4.1. link: http://incompleteideas.net/book/bookdraft2017nov5.pdf

        # to remind the function for writing values of states check side 34 of this powerpoint: http://ai.berkeley.edu/slides/Lecture%208%20--%20MDPs%20I/SP14%20CS188%20Lecture%208%20--%20MDPs%20I.pptx
        for _ in range(iterations): # in python _ is can be a variable name just like a or b or jack.
            # it is usually used when the variable itself is not important to us, but we are only using it to iterate over something
            values_copy = self.values.copy() # copy the values of state, don't change them live

            for state in mdp.getStates():  # go through every state
                if mdp.isTerminal(state): # this is the terminal state
                    pass
                # no update required for this state. go to next state

                # here u should pick the maximum value of all possible Q(s,a)s from this state
                else:
                # get possible actions from this state. it is stored in self.mdp
                    actions = self.mdp.getPossibleActions(state)
                # get all possible Q(s,a)s. you can use self.getQValue function
                    q_values = (self.getQValue(state,action) for action in actions)
                # save the new V(s) with the max( [Q(s,a)] )
                    values_copy[state] = max(q_values)

            # update V(s) of all states with the new V(s) of this iteration. So in each iteration you update V(all states) altogether
            # for state in mdp.getStates():
            #     self.values[state] = values_copy[state]
            self.values = values_copy.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        import types
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # "*** YOUR CODE HERE ***"

        # in order to calculate the Q(state,action), u should see the possible transitions of Q(s,a) and how probable each transition is
        # to remind the function for writing values of states check side 34 of this powerpoint: http://ai.berkeley.edu/slides/Lecture%208%20--%20MDPs%20I/SP14%20CS188%20Lecture%208%20--%20MDPs%20I.pptx

        q_value = 0
        for nextState,prob in self.mdp.getTransitionStatesAndProbs(state,action): # iterate over Q(s,a,s'),P(s,a,a')
            q_value += prob * ( self.mdp.getReward(state,action,nextState) + self.discount*self.values[nextState] ) # update q value




        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # "*** YOUR CODE HERE ***"

        actions = self.mdp.getPossibleActions(state) # get possible actions from this state. it is stored in mdp
        if len(actions) == 0:  # ----- action is empty: (it is terminal state, so there is no possible actions)
            return None
        # for each action check the Q(s,a) and them to a list
        qList = []
        for action in actions:
            qList.append(self.computeQValueFromValues(state,action))
        # choose the action index that leads to maximum q value. you can use np.argmax( ) . u should add "import numpy as np" to your imports
        maxIndex = np.argmax(qList)
        # return that action
        return actions[maxIndex]
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
