class soldierSalaryProblem:

    def __init__(self, soldier_salary, money_bags):
        self.expanded = 0
        soldier_salary.sort(reverse=True)
        money_bags.sort(reverse=True)
        self.soldier_salary = soldier_salary
        self.money_bags = money_bags

    def getStartState(self):
        return (self.soldier_salary, self.money_bags)

    def isGoalState(self, state):
        for i in range(len(state[0])):
            if (state[0][i] > 0):
                return False
        return True

    def getSuccessors(self, state):

        successors = []

        firstPartOfState = state[0]
        secondPartOfState = state[1]

        numberOfSoldier = 0
        status = True
        while status:
            if (firstPartOfState[numberOfSoldier] > 0):
                status = False
            else:
                if (numberOfSoldier == len(firstPartOfState) - 1):
                    return []
                else:
                    numberOfSoldier = numberOfSoldier + 1

        for i in range(len(secondPartOfState)):
            first_part = firstPartOfState[:]
            second_part = secondPartOfState[:]
            action = (numberOfSoldier, secondPartOfState[i])
            first_part[numberOfSoldier] = first_part[numberOfSoldier] - second_part[i]
            second_part.remove(second_part[i])
            successors.append(((first_part, second_part), action))

        self.expanded += 1
        return successors

    def printAnswer(self, path):
        print('answer: ')

        answer = []
        for i in range(len(path)):
            answer.append((self.soldier_salary[path[i][0]], path[i][1]))
        print(answer)

    def randomized_tuple(self):

        import random as rn
        a = []
        for i in range(len(self.money_bags)):
            a.append((rn.randint(0, len(self.soldier_salary) - 1), self.money_bags[i]))
        return a

    def random_final_state(self, randomizedTuple):
        first_part = self.soldier_salary[:]
        second_part = self.money_bags[:]

        for i in range(len(randomizedTuple)):
            value = randomizedTuple[i][1]
            second_part.remove(value)
            first_part[randomizedTuple[i][0]] = first_part[randomizedTuple[i][0]] - value

        return ((first_part, second_part), randomizedTuple)

    def heuristic(self, state):

        sum_need_money = 0
        for i in range(len(state[0][0])):
            if (state[0][0][i] > 0):
                sum_need_money += (state[0][0][i]) **2

        return sum_need_money

    def cost_function(self, state):

        sum_money_payed = 0
        for i in range(len(state[1])):
            sum_money_payed += (state[1][i][1])**2

        return sum_money_payed
