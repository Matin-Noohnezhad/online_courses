def plotData(x, y, c='*', xl=None, yl=None, filename='test.png'):
	from pylab import *

	#PLOTDATA Plots the data points x and y into a new figure 
	#   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
	#   population and profit.

	# ====================== YOUR CODE HERE ======================
	# Instructions: Plot the training data into a figure using the 
	#               "figure" and "plot" commands. Set the axes labels using
	#               the "xlabel" and "ylabel" commands. Assume the 
	#               population and revenue data have been passed in
	#               as the x and y arguments of this function.
	#
	# Hint: You can use the 'rx' option with plot to have the markers
	#       appear as red crosses. Furthermore, you can make the
	#       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

	
	
	plot(x, y, c)
	xlabel(xl);
	ylabel(yl);
	grid(True)
	savefig(filename)
	show()

	# ============================================================

# x = range(0, 100)
# y = [i*i for i in x]
# plotData(x, y)