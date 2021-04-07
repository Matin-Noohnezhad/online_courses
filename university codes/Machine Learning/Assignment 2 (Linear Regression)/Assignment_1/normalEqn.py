import numpy as np 
def normalEqn(X, y):
	#NORMALEQN Computes the closed-form solution to linear regression 
	#   NORMALEQN(X,y) computes the closed-form solution to linear 
	#   regression using the normal equations.
	print (X.shape)
	theta = np.zeros((X.shape[1], 1))
	# ====================== YOUR CODE HERE ======================
	# Instructions: Complete the code to compute the closed form solution
	#               to linear regression and put the result in theta.
	#
	xtx = np.dot(np.transpose(X) , X)
	xtxInverse = np.linalg.inv(xtx)
	xty = np.dot(np.transpose(X) , y)
	theta = np.dot(xtxInverse,xty)
	# -------------------------------------------------------------


	# ============================================================
	return theta 
