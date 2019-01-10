import numpy as np


#-----------------------------GAUSSIAN STUFF----------------------------------#
def gaussian_function_2D(x, y, sigma=1):
    return np.e ** (-((x ** 2) + (y ** 2)) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def gaussian_2D(n, sigma=1):
    kernel = np.ones((n, n))
    vals = list(range(int(-n/2), int(n/2)+1))
    for x in range(n):
        for y in range(n):
            kernel[x][y] = gaussian_function_2D(vals[x], vals[y], sigma)
    return kernel

#-----------------------------NOMRMALISATION---------------------------------#
def normalise(vector):
    norm = np.sqrt(np.sum(vector**2))
    if norm == 0:
        return vector
    return vector / norm