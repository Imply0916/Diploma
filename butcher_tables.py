import numpy as np


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EXPLICIT METHODS:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def ForwardEuler():
    A = np.array([0])
    b = np.array([1])
    c = np.array([0])
    return A, b, c


def ExplicitMidpointMethod():
    A = np.array([
        [0, 0], 
        [1, 0]
    ])
    b = np.array([1./2., 1./2.])
    c = np.array([0, 1.])
    return A, b, c


def KuttaThirdOrderMethod():
    A = np.array([
        [0, 0, 0], 
        [1./2., 0, 0],
        [-1., 2., 0]
    ])
    b = np.array([1./6., 2./3., 1./6.])
    c = np.array([0, 1./2., 1.])
    return A, b, c



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPLISIT METHODS:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def GaussLegendreSixOrder():  # order 6 
    A = np.array([
        [5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30],
        [ 5/36 + np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24],
        [ 5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]
    ])
    b = np.array([5/18, 4/9, 5/18])
    c = np.array([1/2 - np.sqrt(15)/10, 1/2, 1/2 + np.sqrt(15)/10])

    return A, b, c


def CrankNicolsonMethodSecondOrder():  # order 2 
    A = np.array([
        [0, 0], 
        [1./2., 1./2.]
    ])
    b = np.array([1/2, 1/2])
    c = np.array([0, 1])

    return A, b, c
    


# Diagonally Implicit Runge–Kutta 
def DIRKThirdOrder():  # order 4
    A = np.array([
        [1/2, 0, 0, 0], 
        [1/6, 1/2, 0, 0], 
        [-1/2, -1/2, 1/2, 0],
        [3/2, -3/2, 1/2, 1/2],
    ])
    b = np.array([3/2, -3/3, 1/2, 1/2])
    c = np.array([1/2, 2/3, 1/2, 1])

    return A, b, c


# Diagonally Implicit Runge–Kutta 
def DIRKFourOrder():  # order 4
    A = np.array([
        [1/4, 0, 0, 0, 0], 
        [1/2, 1/4, 0, 0, 0], 
        [17/50, -1/25, 1/4, 0, 0],
        [371/1360, -137/2720, 15/544, 1/4, 0],
        [25/24, -49/48, 125/16, -85/12, 1/4]
    ])
    b = np.array([25/24, -49/48, 125/16, -85/12, 1/4])
    c = np.array([1/4, 3/4, 11/20, 1/2, 1])

    return A, b, c



