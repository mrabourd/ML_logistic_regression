import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR


X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])
thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
mylr = MyLR(thetas)
# Example 0:
print(mylr.predict_(X))
# Output:
# array([[0.99930437],
# [1.
# ],
# [1.
# ]])


# Example 1:
print(mylr.loss_(X,Y))
# Output:
# 11.513157421577002


# Example 2:
print("fit ex2:", mylr.fit_(X, Y))
print("theta:", mylr.theta)
# # Output:
# # array([[ 2.11826435]
# # [ 0.10154334]
# # [ 6.43942899]
# # [-5.10817488]
# # [ 0.6212541 ]])


# # Example 3:
print("predict ex3:", mylr.predict_(X))
# # Output:
# # array([[0.57606717]
# # [0.68599807]
# # [0.06562156]])


# # Example 4:
print("loss ex4:", mylr.loss_(X,Y))
# Output:
# 1.4779126923052268