#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:30:59 2021

@author: Alice
"""
import os
import black

from scipy.spatial import distance
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

""" 
Machine Learning
ASG 4
Alice Hamberger

For each function I have put an example run of the function."""

"""
The distance1 function shows a graph that exhibits exponential growth of
the average distance between points for increasing dimensions. The standard
deviation signified by errror bars also increases for increasing dimensions.
This makes sense, because as the number of dimensions increase, there is 
a smaller probability that two points have similar coordinates for all
dimensions. Two points need only to be far apart in one dimension out of 
n-dimensions to be far apart in general.

For the distance1 funciton below, I first wrote longer code, because I thought
that I needed to remove the duplicates of distance between points, as if they
are indistance matrix, every distance appears twice. I resolved this by nulling 
the lower triangle, so I would have only half the data. This code is shown
below. After some thinking, I realized that I can just average the entire
matrix, because if every value is duplicated  then this does not affect the
average because all values still have the same weight.  After this realization
I implemented way shorter code (the one liner 'distanceslist'). Suprisingly,
the error bars for the remove duplicates method seem to be a bit smaller 
on average.

Previous (remove duplicates) code:
    
        take out distanceslist
        distancedf = pd.DataFrame(distanceformula.pairwise(points))
        triangledf = pd.DataFrame(np.triu(distancedf))
        nandf = triangledf.replace(0, np.NaN)

        lists = nandf.values.tolist()
        distanceslist = [item for sublist in lists for item in sublist]
"""


def distance1(p, n):
    """ Input: an integer (p) of the number of random uniform points
        and an integer (n) of the number of dimensions. 
        Retruns: a plot where the x-axis is the number of dimensions (n) in a 
        log scale, and the y-axis is the average distance between points. The
        error bars are the standard deviation. """

    meanl = []
    stdl = []

    for x in range(2, n + 1):
        points = np.random.uniform(-1, 1, (p, x))
        distanceformula = DistanceMetric.get_metric("euclidean")
        distanceslist = pd.DataFrame(distanceformula.pairwise(points)).stack()

        mean = np.nanmean(distanceslist)
        std = np.nanstd(distanceslist)
        meanl.append(mean)
        stdl.append(std)

    xaxis = np.arange(2, n + 1)
    fig, ax = plt.subplots()
    ax.bar(xaxis, meanl, yerr=stdl, align="center", alpha=0.5, capsize=10)
    ax.set_xticks(xaxis)
    ax.set_xticklabels(xaxis)
    ax.set_title("Average Distance between Points in an n-dimensional Hypercube")
    ax.set_ylabel("Average Distance between Points")
    ax.set_xlabel("Dimensions")
    plt.xscale("log")


distance1(100, 100)


# 2
"""
I had a lot of print statement in my code to see what was going on but I
removed those to make the code cleaner. I could have written distorigin in
one line as well but I decided against it for clarity. I left one print
statement in incircle to show you how I had the code for all functions 
before. However, this print statement got messy when calling these functions
for other things like the graphs r and s.
"""


def distorigin(p, n):
    """ Input: number of points (p) and number of dimensions (n).  
        Returns: numpy array of each points distance to origin in n dimensions."""
    points = np.random.uniform(-1, 1, (p, n))
    origin = np.zeros((1, n))
    disto = distance.cdist(points, origin)
    return disto.ravel()


print("each points distance to origin in n-dimensions (example):", distorigin(5, 2))

# a
def incircle(p, n):
    """ Input: number of points (p) and number of dimensions (n). 
        Returns: fraction of points inside n-dimensional hypersphere."""
    # print('fraction of points in',n,'-dimensional hypersphere:',output)
    return np.count_nonzero(distorigin(p, n) <= 1) / p


print("fraction of points in n-dimensional hypersphere (example):", incircle(5, 4))


# b


def avdistorigin(p, n):
    """ Input: number of points (p) and number of dimensions (n). 
        Returns: average distance (r) of points from origin 
        in an n-dimensional hypersphere."""
    points = np.array([])
    for x in distorigin(p, n):
        if x <= 1:
            points = np.append(points, x)
    return np.mean(points)


print(
    "average distance of points to origin in n-dimensional hypersphere (example):",
    avdistorigin(5, 2),
)


"""
Same Function but with conversion to list, because it took me an our to figure out
that in numpy you have to equal your array  to your changed array because it 
makes a copy rather than changing it.

def avdistorigin (p,n):
        Input: number of points (p) and number of dimensions (n). 
        Returns: average distance (r) of points from origin 
        in an n-dimensional hypersphere.
    points = []
    for x in distorigin(p,n).tolist():
        if x<=1:
            points.append(x)
    return np.mean(points)
"""

# c
def inshell(p, n):
    """ Input: number of points (p) and number of dimensions (n). 
        Returns: fraction of points (s) inside shell of n-dimensional hypersphere.
        The shell is defined as radius 0.99<=shell<=1.00 centered at origin."""
    return np.count_nonzero((0.99 <= distorigin(p, n)) & (distorigin(p, n) <= 1)) / p


print(
    "fraction of points in shell of n-dimensional hypersphere (example):",
    inshell(100, 2),
)

# d see above

# e


def graphr(p, n):

    """ Input: number of points (p) and number of dimensions (n). 
         Return: A graph where the x-axis is the number of n-dimensions
         on a logarithimc scale and the y-axis is the average distance of points 
         to origin (r) in a n-dimensional hypersphere."""
    meanl = []
    for x in range(2, n + 1):
        meanl.append(avdistorigin(p, x))

    xaxis = np.arange(2, n + 1)
    fig, ax = plt.subplots()
    ax.bar(xaxis, meanl, align="center", alpha=0.5)
    ax.set_xticks(xaxis)
    ax.set_xticklabels(xaxis)
    ax.set_title(
        "Average Distance of Points from Origin (r) in an n-dimensional Hypersphere"
    )
    ax.set_ylabel("Average Distance of Points from Origin (r)")
    ax.set_xlabel("Dimensions")
    plt.xscale("log")


graphr(100, 400)

"""
The number of points that fall into the shell in n-dimensions is suprisingly
high for up to around 8 dimensions, but already steadily decreasing when 
the number of dimensions increases. After approximately 10 dimensions there
are consistently 0 points that fall into the shell. This makes sense, because
as the number of dimensions increases, there is a lower probability that the 
distances from the origin for one point in all dimensions is in this small
shell range. 
"""


def graphs(p, n):

    """ Input: number of points (p) and number of dimensions (n). 
         Return: A graph where the x-axis is the number of n-dimensions
         on a logarithimc scale and the y-axis is the average distance of points 
         to origin (r) in a n-dimensional hypersphere."""
    meanl = []
    for x in range(2, n + 1):
        meanl.append(inshell(p, x))

    xaxis = np.arange(2, n + 1)
    fig, ax = plt.subplots()
    ax.bar(xaxis, meanl, align="center", alpha=0.5)
    ax.set_xticks(xaxis)
    ax.set_xticklabels(xaxis)
    ax.set_title("Fraction of Points that fall into shell (s) in n-dimenions")
    ax.set_ylabel("Fraction of Points that fall into shell (s)")
    ax.set_xlabel("Dimensions")
    plt.xscale("log")


graphs(10, 100)
