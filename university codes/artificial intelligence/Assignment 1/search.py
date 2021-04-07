# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



def generic_search_uninformed(problem, fringe):
    ------ = ---- # empty set of explored nodes
    path = ------ # empty list of actions
    start_state = problem.getStartState() # (x,y) position
    -------- = (-----,-----)  # start node , actions 
    ------- # push (start state , actions) to fringe

    while -----: # while fringe is not empty
        (----, ----) = ---- # pop a successor from fringe

        if ----- :# check if current node is the goal node
            return ----- # return sequence of actions from start to goal
        if ----: # if the node poped from fringed is NOT already explored 
            ------ # add it to explored set   

            for ------,------ in problem.getSuccessors(----): # and get its children : list of nodes and actions that lead to those nodes
                new_path = ----- + ----- # current path + new action
                ----- = (-----, -----) # tuple of state and path
                fringe.push(------) 
                
                
def generic_search_informed(problem, fringe,heuristic):
    ------ = ---- # empty set of explored nodes
    path = ------ # empty list of actions
    start_state = problem.getStartState() # (x,y) position
    gn = # g(start state)
    hn = # h(start state)
    cost = ----- # cost at the begining
    -------- = (-----,-----)  # start node , actions , 
    ------- # push (start state , actions) to fringe

    while -----: # while fringe is not empty
        (----, ----) = ---- # pop a successor from fringe

        if ----- :# check if current node is the goal node
            return ----- # return sequence of actions from start to goal
        if ----: # if the node poped from fringed is NOT already explored 
            ------ # add it to explored set   

            for ------,------ in problem.getSuccessors(----): # and get its children : list of nodes and actions that lead to those nodes
                new_path = ----- + ----- # current path + new action
                ----- = (-----, -----) # tuple of state and path
                ----- = ------ + ------   # g(new state) = cost till now + cost of new action
                ----- = ------ # h(new state) = heuristic of new state ** for USC u can use the nullHeuristic function
                new_cost = # g(new state) + h(new state)
                fringe.push(------,-----) # check the push function in util.priorityQueue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    fringe = util.----- # what strategy is required for  DFS ? LIFO or FIFO? stack or queue ?
    return generic_search_uninformed(problem, fringe, add_to_fringe_fn)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.----- # what strategy is required for  BFS ? LIFO or FIFO ? stack or queue?
    return generic_search_uninformed(problem, fringe)

def uniformCostSearch(problem):  
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.----- #  check priorityQueue() in util.py
    return generic_search_informed(problem, fringe, ----) # USC is just like A* if you use heuristic = 0 cosntatnt function
    
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def my_heuristic(state,problem=None):
    
    "*** YOUR CODE HERE ***"
    
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    return generic_search(problem, fringe)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
