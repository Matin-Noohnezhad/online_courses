import util


class Search:

    def __init__(self):
        self.max_fringe_size = 0

    def depth_first_search(self, problem):
        # print(problem.getStartState())
        # print(problem.isGoalState(problem.getStartState()))
        # print(problem.getSuccessors(problem.getStartState()))

        node = (problem.getStartState(), [])
        #
        fringe = util.Stack()
        fringe.push(node)
        #
        while True:
            # print('next')
            if fringe.isEmpty():
                return []
            if (self.max_fringe_size < len(fringe.list)):
                self.max_fringe_size = len(fringe.list)
            current_node = fringe.pop()
            path = current_node[1]
            if (problem.isGoalState(current_node[0])):
                return path
            for successor in problem.getSuccessors(current_node[0]):
                new_path = path[:]
                new_path.append(successor[1])
                fringe.push((successor[0], new_path))

    def back_tracking_search(self, problem):
        # print(problem.getStartState())
        # print(problem.isGoalState(problem.getStartState()))
        # print(problem.getSuccessors(problem.getStartState()))

        node = (problem.getStartState(), [])
        #
        fringe = util.Stack()
        fringe.push(node)
        #
        while True:
            # print('next')
            if fringe.isEmpty():
                return []
            if (self.max_fringe_size < len(fringe.list)):
                self.max_fringe_size = len(fringe.list)
            current_node = fringe.pop()
            path = current_node[1]
            if (problem.isGoalState(current_node[0])):
                return path
            for successor in problem.getSuccessors(current_node[0]):
                new_path = path[:]
                new_path.append(successor[1])
                #
                sum_soldier_salaries = 0
                for element in successor[0][0]:
                    if (element > 0):
                        sum_soldier_salaries += element
                #
                if (sum(successor[0][1]) >= sum_soldier_salaries):
                    fringe.push((successor[0], new_path))

    def iterative_min_conflict(self, problem, roof):

        import random as rn
        self.counter_for_IMC = 0
        state = problem.random_final_state(problem.randomized_tuple())

        for i in range(roof):
            if (problem.isGoalState(state[0])):
                return state[1]

            random_index = rn.randint(0, len(state[1]) - 1)

            max_index = state[1][random_index][0]
            max_util = 0

            for j in range(len(state[0][0])):
                util = 0
                if (j != state[1][random_index][0]):
                    self.counter_for_IMC += 1
                    value = state[1][random_index][1]
                    previous_index_previous_value = state[0][0][state[1][random_index][0]]
                    previous_index_next_value = state[0][0][state[1][random_index][0]] + value
                    next_index_previous_value = state[0][0][j]
                    next_index_next_value = state[0][0][j] - value
                    util += (abs(previous_index_previous_value) - abs(previous_index_next_value))
                    util += (abs(next_index_previous_value) - abs(next_index_next_value))
                    if (previous_index_previous_value <= 0):
                        util -= 50
                    if (previous_index_next_value <= 0):
                        util += 50
                    if (next_index_previous_value <= 0):
                        util -= 50
                    if (next_index_next_value <= 0):
                        util += 50
                    if (util > max_util):
                        max_util = util
                        max_index = j
            if (max_util > 0):
                new_index = max_index
                previous_index = state[1][random_index][0]
                value = state[1][random_index][1]
                state[1].remove((previous_index, value))
                state[1].append((new_index, value))
                state[0][0][new_index] -= value
                state[0][0][previous_index] += value
            if (i == (roof - 1)):
                return []

    def a_star_search(self, problem):
        # print(problem.getStartState())
        # print(problem.isGoalState(problem.getStartState()))
        # print(problem.getSuccessors(problem.getStartState()))

        node = (problem.getStartState(), [])
        #
        fringe = util.PriorityQueue()
        fringe.push(node, 0)
        #
        while True:
            # print('next')
            if fringe.isEmpty():
                return []
            if (self.max_fringe_size < len(fringe.heap)):
                self.max_fringe_size = len(fringe.heap)
            current_node = fringe.pop()
            path = current_node[1]
            if (problem.isGoalState(current_node[0])):
                return path
            for successor in problem.getSuccessors(current_node[0]):
                new_path = path[:]
                new_path.append(successor[1])
                fringe.push((successor[0], new_path),
                            problem.cost_function((successor[0], new_path)) + problem.heuristic(
                                (successor[0], new_path)))
