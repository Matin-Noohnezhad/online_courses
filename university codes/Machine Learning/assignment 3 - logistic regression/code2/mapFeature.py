import numpy as np
def mapFeature(X1, X2):
	# MAPFEATURE Feature mapping function to polynomial features
	#
	#   MAPFEATURE(X1, X2) maps the two input features
	#   to quadratic features used in the regularization exercise.
	#
	#   Returns a new feature array with more features, comprising of 
	#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
	#
	#   Inputs X1, X2 must be the same size
	#

	degree = 6
	out = np.ones((X1.shape[0], 3))
	#...
	#...
	return out