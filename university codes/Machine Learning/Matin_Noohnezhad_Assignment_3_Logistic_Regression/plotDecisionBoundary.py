from pylab import *
from mapFeature import mapFeature
from plotData import *


def plotDecisionBoundary(theta, X, y):
    # PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    # the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    #   positive examples and o for the negative examples. X is assumed to be 
    #   a either 
    #   1) Mx3 matrix, where the first column is an all-ones column for the 
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    # print X
    # plot(X[:,1:2], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X[:, 1]) - 2, max(X[:, 1]) + 2]

        # Calculate the decision boundary line
        plot_y = [(-1.0 / theta[2]) * (theta[1] * x + theta[0]) for x in plot_x]

        # Plot, and adjust axes for better viewing
        import plotData as pd
        pd.plotData(X[:, [1, 2]], y, x_db=plot_x, y_db=plot_y, xl='Exam 1 score', yl='Exam 2 score')
        # plot(plot_x, plot_y)
        # grid(True)
        # show()

        # Legend, specific for the exercise
        # legend('Admitted', 'Not admitted', 'Decision Boundary')
        # axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        # U, V = np.meshgrid(u, v)
        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        print('theta:')
        print(theta)
        for i in range(len(u)):
            for j in range(len(v)):
                z[i][j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]])), np.reshape(theta, (len(theta), 1)))
        z = z.transpose()
        # important to transpose z before calling contour
        # print(z)
        # print(u)
        # print(v)
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        # plt.contour(u, v, z,[0])#, [0, 0], 'LineWidth', 2)
        plotData(X[:, [1, 2]], y, u=u, v=v, z=z, xl='Exam 1 score', yl='Exam 2 score',filename="test2.png")
        plt.show()
