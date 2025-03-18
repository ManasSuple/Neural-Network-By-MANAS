import numpy as np

inputs = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
bias = [2.0, 3.0, 0.5]
# We can transpose the list in python so we have to conver weight matrix into array first so we did it by using np.array(weights).T
output = np.dot(inputs, np.array(weights).T) + bias
print(output)