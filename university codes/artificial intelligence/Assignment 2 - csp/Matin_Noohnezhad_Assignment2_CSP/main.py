import searchFunctions as sf
import Problem as pb

arr = [14, 7, 5, 8, 10, 4, 11]
arr2 = [8, 5, 4, 6, 9, 2, 6, 9, 4, 4, 6, 2]
# part 1 (depth-first search)
print('##########################################################')
soldier_salary = arr[:]
money_bags = arr2[:]

ssp = pb.soldierSalaryProblem(soldier_salary, money_bags)
search = sf.Search()

path = search.depth_first_search(ssp)

print('**depth-first search**')
ssp.printAnswer(path)
print('number of expanded node: ' + str(ssp.expanded))
print('maximum size of fringe: ' + str(search.max_fringe_size))
# # part 2 (back-tracking search)
print('##########################################################')
soldier_salary = arr[:]
money_bags = arr2[:]

ssp = pb.soldierSalaryProblem(soldier_salary, money_bags)
search = sf.Search()

path = search.back_tracking_search(ssp)

print('**back-tracking search**')
ssp.printAnswer(path)
print('number of expanded node: ' + str(ssp.expanded))
print('maximum size of fringe: ' + str(search.max_fringe_size))
# # part 3 (iterative min-conflict)
print('##########################################################')
soldier_salary = arr[:]
money_bags = arr2[:]

ssp = pb.soldierSalaryProblem(soldier_salary, money_bags)
search = sf.Search()
numberOfValueChanging = 300000
path = search.iterative_min_conflict(ssp, numberOfValueChanging)
print('**Iterative min-conflict**')
if (len(path) == 0):
    print("Iterative min-conflict couldn't find answer this time.(within the ", str(numberOfValueChanging), ")")
else:
    ssp.printAnswer(path)
print('number of checking for changing the value of a variable: ' + str(search.counter_for_IMC))
print(
    '^^^^note:Most of the iterative min-conflict doesnt give you an answer(for example 4 of 5 times with this example)^^^^')
# part 4 (A* search)
print('##########################################################')
soldier_salary = arr[:]
money_bags = arr2[:]

ssp = pb.soldierSalaryProblem(soldier_salary, money_bags)
search = sf.Search()

path = search.a_star_search(ssp)

print('**a-star search**')
ssp.printAnswer(path)
print('number of expanded node: ' + str(ssp.expanded))
print('maximum size of fringe: ' + str(search.max_fringe_size))
print('##########################################################')
