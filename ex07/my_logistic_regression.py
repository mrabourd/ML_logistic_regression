import numpy as np

class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta


    def sigmoid_(self, x):
        """
        Compute the sigmoid of a vector.
        Args:
        x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or \
            x.size == 0:
            return None
        sigmoid = 1 / (1 + np.exp(-1 * x))
        return sigmoid
        

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or\
            not isinstance(self.theta, np.ndarray) or \
            x.size == 0 or self.theta.shape[1] != 1:
            return None

        m = x.shape[0]
        n = x.shape[1]

        x_one = np.ones((m, 1))
        x_hat = np.column_stack((x_one, x))

        # y_hat = np.zeros((m, ))

        pred = x_hat @ self.theta
        y_hat = self.sigmoid_(pred)
        # print("y_hat  predict:", y_hat)
        return y_hat.reshape(-1, 1) 


    
    def vec_log_loss_(self, y, y_hat, eps=1e-15):
        """
        Computes the logistic loss value.
        Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: epsilon (default=1e-15)
        Returns:
        The logistic loss value as a float.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None

        m = y.shape[0]
        one_vec = np.ones((m, 1))

        loss = (y * np.log(y_hat + eps)) + (one_vec - y) * np.log(one_vec - y_hat + eps)
        
        res = -1 / m * np.sum(loss)
        res_format = round(float(res), 17)
        return res_format

    def loss_(self, x, y):
        """
        Computes the logistic loss value for given x and y.
        Args:
        x: has to be a numpy.ndarray, a vector of dimension m * n.
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        Returns:
        The logistic loss value as a float.
        None on any error.
        """
        print("x", x)
        y_hat = self.predict_(x)
        return self.vec_log_loss_(y, y_hat)
    

    def vec_log_gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatibl
        Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or \
            not isinstance(self.theta, np.ndarray) or \
            x.shape[0] != y.shape[0] or \
            self.theta.shape[1] != 1:
            return None
        
        m = x.shape[0]
        y_hat = self.predict_(x)

        gradient = np.zeros((x.shape[1] + 1, 1))
        x_hat = np.column_stack((np.ones((m, 1)), x))
        x_t = x_hat.T
        
        gradient = (1/m) * (x_t @ (y_hat - y))
        return gradient
        

    def fit_(self, x, y):
        """
        Fits the logistic regression model to the data.
        Args:
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        Returns:
        The updated theta values as a numpy.ndarray.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        """
        if not isinstance(x, np.ndarray) or\
            not isinstance(y, np.ndarray) or \
            x.size == 0 or y.size == 0:
            return None

        y = y.reshape(y.shape[0], 1)

        for _ in range(self.max_iter):
            gradient = self.vec_log_gradient(x, y)
            if gradient is not None:
                self.theta -= self.alpha * gradient

        return self.theta