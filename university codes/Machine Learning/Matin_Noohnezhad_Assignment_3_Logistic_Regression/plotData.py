from pylab import *


def plotData(x, y, c='*', xl=None, yl=None, filename='test.png', x_db=None, y_db=None, u=[], v=[], z=[]):
    # PLOTDATA Plots the data points x and y into a new figure
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

    # plot(x, y, c)

    zero = y == 0
    one = y == 1
    # print(y.shape)
    # print(zero.shape)
    # print(x[:,0].shape)
    # print(type(y))
    x_0 = x[:, 0]
    x_0 = x_0.reshape(len(x_0), 1)
    x_1 = x[:, 1]
    x_1 = x_1.reshape(len(x_1), 1)
    plot(x_0[zero], x_1[zero], 'o', mfc='none')
    plot(x_0[one], x_1[one], '+')
    xlabel(xl)
    ylabel(yl)
    if (x_db != None and y_db != None):
        plot(x_db, y_db)
    if(len(u)!=0 and len(v)!=0 and len(z)!=0):
        contour(u,v,z,[0])
    grid(True)
    savefig(filename)
    show()

# ============================================================

# x = range(0, 100)
# y = [i*i for i in x]
# plotData(x, y)
