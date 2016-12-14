# PHYS220 Final Project

**Author(s):** Aaron Grisez

[![Build Status](https://travis-ci.com/chapman-phys220-2016f/final-aarongrisez.svg?token=NKnMzaV57yuvZyF9zLxy&branch=master)](https://travis-ci.com/chapman-phys220-2016f/final-aarongrisez)

**Due date:** 2016/12/13

## Specification

1. You have one week to complete your final. Work alone.
1. Read ```final-rubric.pdf``` and follow the instructions.

## Final Description

The assignment is to analyze the Rossler Attractor. We will use numerical integration methods and various visualizations to understand the chaotic behavior that arises from the nonlinearity of the governing Ordinary Differential Equations. This investigation will only look at varying a single parameter in these equations, leaving the others constant. All of the code to generate the data and plots are included. Pre-rendered images and calculated data are included in the folders named "images" and "data". A few of those pre-rendered data sets are used directly in the notebook, but the code to generate them is un-altered. There is a description for how to generate your own data sets and plots below.

### Data Sets
The solutions read into the notebook via the Pandas interface can be recreated using the rossler() function in rossler.py. The only required argument for rossler() is the $c$ parameter which this investigation is tuning.

### Plots
The 3-dimensional plots were generated using the plotxyz() function in rossler.py. It takes 2 parameters, the solution dataframe as described above and a time index for ignoring transient behavior. All points before the time index will be ignored in the plot.

## Assessment

I really enjoyed this assignment, particularly since it was a nice extension of the midterm assignment. It was very informative (and fun) to get some experience with both Pandas and Numba.

## Honor Pledge

I pledge that all the work in this repository is my own with only the following exceptions:

* Content of starter files supplied by the instructor;
* Code borrowed from another source, documented with correct attribution in the code and summarized here.

Signed,

Aaron Grisez
