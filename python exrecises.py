import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from os import chdir, getcwd

def add(x, y):
    """Add two numpy arrays elementwise"""
    return np.add(x,y)

def mult(x, y):
    """Multiply two numpy arrays elementwise"""
    return np.multiply(x,y)

def sort_series(s):
    """Sort a pandas series"""
    return s.sort_values()

def augment_series(series, extra):
    """Augments a series with extra data"""
    return series.append(pd.Series(extra))


def series_stats(s):
    """Return the mean and standard deviation of the series"""
    return (s.mean(), s.std())

def series_diff(s1, s2):
    x = list(set(s1)-set(s2))
    return pd.Series(x)
    
            
def series_counts(s):
    """Returns the number of occurrences of each element of s"""
    return s.value_counts()

def series_diffs(s):
    """Returns a series of the differences between adjacent elements of the series"""
    return s.diff()

def series_parse_dates(s):
    """Convert each date string in s into a datetime structure"""
    return pd.to_datetime(s)

       
def filter_multiple_vowels(s):
    """Drop from series any string which contains more than one vowel"""
    n = []
    for string in s:
        count = 0
        
        for letter in string:
            if letter in "aeiouAEIOU":
                count += 1

        if count < 2:
            n.append(string)

    return pd.Series(n)


def timeseries_sundays(year):
    """Return a TimeSeries containing all of the Sundays of the given year"""
    return pd.date_range(start=str(year), end=str(year+1), freq="W-SUN")


def frame_occ_stats(dataframe):
    """Return a salary series containing averages for each occup in dataframe"""
    avg = dataframe.groupby('occ').mean()
    salaryl = avg['salary'].tolist()
    occl = avg.index.values.tolist()
    return pd.Series(salaryl,name='salary', index= occl)
    

def frame_fill_nans(dataframe, val):
    """Replace all NaNs in the dataframe with val"""
    return dataframe.fillna(val)
