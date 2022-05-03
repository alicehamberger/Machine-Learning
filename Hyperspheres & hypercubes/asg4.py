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




def distance1(p,n):
    """ Input: an integer (p) of the number of random uniform points
        and an integer (n) of the number of dimensions. 
        Retruns: a plot where the x-axis is the number of dimensions (n) in a 
        log scale, and the y-axis is the average distance between points. The
        error bars are the standard deviation. """

    meanl = []
    stdl = []
    
    for x in range(2,n+1):
        points = np.random.uniform(-1,1,(p,x)) 
        distanceformula = DistanceMetric.get_metric('euclidean') 
        distanceslist = pd.DataFrame(distanceformula.pairwise(points)).stack()
        
        mean = np.nanmean(distanceslist)
        std = np.nanstd(distanceslist)
        meanl.append(mean)
        stdl.append(std)

    xaxis = np.arange(2,n+1)
    fig, ax = plt.subplots()
    ax.bar(xaxis, meanl, yerr=stdl, align='center', alpha=0.5, capsize=10)
    ax.set_xticks(xaxis)
    ax.set_xticklabels(xaxis)
    ax.set_title('Average Distance between Points in an n-dimensional Hypercube')
    ax.set_ylabel('Average Distance between Points')
    ax.set_xlabel('Dimensions')
    plt.xscale("log")


distance1(100,100)

"""
        take out distanceslist
        distancedf = pd.DataFrame(distanceformula.pairwise(points))
        triangledf = pd.DataFrame(np.triu(distancedf))
        nandf = triangledf.replace(0, np.NaN)

        lists = nandf.values.tolist()
        distanceslist = [item for sublist in lists for item in sublist]
"""
        
# 2

def distorigin(p,n):
    """ Input: number of points (p) and number of dimensions (n).  
        Returns: numpy array of each points distance to origin in n dimensions."""
    points = np.random.uniform(-1,1,(p,n)) 
    origin = np.zeros((1, n))
    disto = distance.cdist(points,origin)
    return disto.ravel()

print('each points distance to origin in n-dimensions (example):',distorigin(5,2))

#a
def incircle(p,n):
    """ Input: number of points (p) and number of dimensions (n). 
        Returns: fraction of points inside n-dimensional hypersphere."""
    #print('fraction of points in',n,'-dimensional hypersphere:',output)
    return np.count_nonzero(distorigin(p,n)<=1) / p

print('fraction of points in n-dimensional hypersphere (example):',incircle(5,4))


#b


def avdistorigin (p,n):
    """ Input: number of points (p) and number of dimensions (n). 
        Returns: average distance (r) of points from origin 
        in an n-dimensional hypersphere."""
    points = np.array([])
    for x in distorigin(p,n):
        if x<=1:
            points = np.append(points,x)
            #print(points)
    return np.mean(points)

print ("average distance of points to origin in n-dimensional hypersphere (example):"
       ,avdistorigin(5,2))


"""
Same Function but wconversion to list, because it took me an our to figure out
that in numpy you have to equal your array  to your changed array because it 
makes a copy rather than changing it

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
#c
def inshell(p,n):
    """ Input: number of points (p) and number of dimensions (n). 
        Returns: fraction of points (s) inside shell of n-dimensional hypersphere.
        The shell is defined as radius 0.99<=shell<=1.00 centered at origin."""
    return np.count_nonzero((0.99<=distorigin(p,n)) & (distorigin(p,n)<=1)) / p

print('fraction of points in shell of n-dimensional hypersphere (example):',inshell(100,2))

#d see above
 
#e 
 
def graphr(p,n):

    """ Input: number of points (p) and number of dimensions (n). 
         Return: A graph where the x-axis is the number of n-dimensions
         on a logarithimc scale and the y-axis is the average distance of points 
         to origin (r) in a n-dimensional hypersphere."""
    meanl = []
    for x in range(2,n+1):
        meanl.append(avdistorigin(p,x))

    xaxis = np.arange(2,n+1)
    fig, ax = plt.subplots()
    ax.bar(xaxis, meanl, align='center', alpha=0.5)
    ax.set_xticks(xaxis)
    ax.set_xticklabels(xaxis)
    ax.set_title('Average Distance of Points from Origin (r) in an n-dimensional Hypersphere')
    ax.set_ylabel('Average Distance of Points from Origin (r)')
    ax.set_xlabel('Dimensions')
    plt.xscale("log")
 
graphr(10,300)

def graphs(p,n):

    """ Input: number of points (p) and number of dimensions (n). 
         Return: A graph where the x-axis is the number of n-dimensions
         on a logarithimc scale and the y-axis is the average distance of points 
         to origin (r) in a n-dimensional hypersphere."""
    meanl = []
    for x in range(2,n+1):
        meanl.append(inshell(p,x))

    xaxis = np.arange(2,n+1)
    fig, ax = plt.subplots()
    ax.bar(xaxis, meanl, align='center', alpha=0.5)
    ax.set_xticks(xaxis)
    ax.set_xticklabels(xaxis)
    ax.set_title('Fraction of Points that fall into shell (s) in n-dimenions')
    ax.set_ylabel('Fraction of Points that fall into shell (s)')
    ax.set_xlabel('Dimensions')
    plt.xscale("log")
 
graphs(10,100)
