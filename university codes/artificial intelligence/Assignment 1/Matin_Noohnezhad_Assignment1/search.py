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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    node = (problem.getStartState(), [], 0)
    #
    fringe = util.Stack()
    fringe.push(node)
    #
    explored = set()
    #
    while True:
        if fringe.isEmpty():
            return []
        current_node = fringe.pop()
        explored.add(current_node[0])
        path_cost = current_node[2]
        path = current_node[1]
        if (problem.isGoalState(current_node[0])):
            return path
        for n in problem.getSuccessors(current_node[0]):
            if (n[0] not in explored):
                new_path = path[:]
                new_path.append(n[1])
                n_cost = path_cost + n[2]
                fringe.push((n[0], new_path, n_cost))

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    start_space = problem.getStartState()

    if (isinstance(start_space[0], tuple)):

        node = (start_space[0], [], 0, start_space[1])
        #
        fringe = util.Queue()
        fringe.push(node)
        #
        explored = set()
        #
        while True:
            if fringe.isEmpty():
                return []
            current_node = fringe.pop()
            explored.add(current_node[0])
            path_cost = current_node[2]
            path = current_node[1]
            if (problem.isGoalState(current_node)):
                return current_node[1]
            if (current_node[0] in current_node[3]):
                current_node[3].remove(current_node[0])
                explored.clear()
                fringe = util.Queue()
            for successor in problem.getSuccessors(current_node):
                if (successor[0] not in explored):
                    new_path = list(path)[:]
                    new_path.append(successor[1])
                    new_cost = path_cost + successor[2]
                    fringe.push((successor[0], new_path, new_cost, current_node[3][:]))

        util.raiseNotDefined()
    else:
        node = (start_space, [], 0)
        #
        fringe = util.Queue()
        fringe.push(node)
        #
        explored = set()
        #
        while True:
            if fringe.isEmpty():
                return []
            current_node = fringe.pop()
            explored.add(current_node[0])
            path_cost = current_node[2]
            path = current_node[1]
            if (problem.isGoalState(current_node[0])):
                return path
            for n in problem.getSuccessors(current_node[0]):
                if (n[0] not in explored):
                    new_path = path[:]
                    new_path.append(n[1])
                    n_cost = path_cost + n[2]
                    fringe.push((n[0], new_path, n_cost))

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    node = (problem.getStartState(), [], 0)
    #
    fringe = util.PriorityQueue()
    fringe.push(node, 1)
    #
    explored = set()
    #
    while True:
        if fringe.isEmpty():
            return []
        current_node = fringe.pop()
        explored.add(current_node[0])
        path_cost = current_node[2]
        path = current_node[1]
        if (problem.isGoalState(current_node[0])):
            return path
        for n in problem.getSuccessors(current_node[0]):
            if (n[0] not in explored):
                new_path = path[:]
                new_path.append(n[1])
                n_cost = path_cost + n[2]
                fringe.push((n[0], new_path, n_cost), n_cost)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    import searchAgents as sa

    start_space = problem.getStartState()

    if (isinstance(start_space[1], list)):
        node = (start_space[0], [], 0, start_space[1])
        #
        fringe = util.PriorityQueue()
        fringe.push(node, 1)
        #
        explored = set()
        #
        number = 0
        while True:
            if fringe.isEmpty():
                return []
            current_node = fringe.pop()
            explored.add(current_node[0])
            path_cost = current_node[2]
            path = current_node[1]
            # print 'current node:',current_node
            if (problem.isGoalState(current_node)):
                return path
            if (current_node[0] in current_node[3]):
                current_node[3].remove(current_node[0])
                explored.clear()
                fringe = util.PriorityQueue()
            for successor in problem.getSuccessors(current_node):
                number = number + 1
                if (successor[0] not in explored):
                    new_path = path[:]
                    new_path.append(successor[1])
                    n_cost = path_cost + successor[2]
                    heuristic = sa.cornersHeuristic((successor[0], current_node[3]), problem)
                    fringe.push((successor[0], new_path, n_cost, current_node[3]), n_cost + heuristic)
                    # print 'number of exploratiooooooooooooooooooon: ', number

        util.raiseNotDefined()
    elif (not isinstance(start_space[0], tuple)):
        node = (start_space, [], 0)
        #
        fringe = util.PriorityQueue()
        fringe.push(node, 1)
        #
        explored = set()
        #
        while True:
            if fringe.isEmpty():
                return []
            current_node = fringe.pop()
            explored.add(current_node[0])
            path_cost = current_node[2]
            path = current_node[1]
            if (problem.isGoalState(current_node[0])):
                return path
            for n in problem.getSuccessors(current_node[0]):
                if (n[0] not in explored):
                    new_path = path[:]
                    new_path.append(n[1])
                    n_cost = path_cost + n[2]
                    heuristic = sa.manhattanHeuristic(n[0], problem)
                    fringe.push((n[0], new_path, n_cost), n_cost + heuristic)
    # astar for foodsearch problem and heuristic
    else:
        # print 'start space: ', problem.getStartState()
        # print 'type of grid: ', problem.getStartState()[1][1][3]
        # print 'successors: ',problem.getSuccessors(problem.getStartState())

        node = (problem.getStartState(), [], 0)
        #
        fringe = util.PriorityQueue()
        fringe.push(node, 1)
        #
        explored = set()
        #
        while True:
            if fringe.isEmpty():
                return []
            current_node = fringe.pop()
            explored.add(current_node[0][0])
            path_cost = current_node[2]
            path = current_node[1]
            if (problem.isGoalState(current_node[0])):
                return path
            for successor in problem.getSuccessors(current_node[0]):
                coeffiecient = 1
                if(successor[0][0] in explored):
                    coeffiecient = coeffiecient*5
                new_cost = path_cost*coeffiecient + successor[2]
                new_path = path[:]
                new_path.append(successor[1])
                fringe.push((successor[0], new_path, new_cost),
                            new_cost + sa.foodHeuristic(current_node[0], problem))


        return

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
