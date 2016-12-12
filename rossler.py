#!usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb

"""
Module to investigate behavior of Rossler oscillator
"""

def rossler(c, T=500.0, dt=0.001):
    """
    Parameters:
    -----------
        c: float
            tunable parameter we are investigating in this module
        T: float
            total time for running the simulation
        dt: float
            mesh spacing for rk4 solutions

    Returns:
    --------
        pd.Dataframe
            4 series pd.Dataframe containing a series of time points and the 3 dimensional solution for the simulation
    """
    n = int(T / dt) #Number of mesh points
    t_vals = np.linspace(0, T, n)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    u_p = np.array([x,y,z])
    u = np.transpose(u_p)
    soln_p = integrate(u, dt, n, c)
    soln = np.transpose(soln_p)
    df = pd.DataFrame({"t":t_vals, "x":soln[0], "y":soln[1], "z":soln[2]})
    indexed_df = df.set_index(t_vals)
    c_string = str(c)
    c_string = c_string.replace('.','_') #Removes periods for filenames
    indexed_df.to_csv('c_equals_' + c_string)
    return indexed_df

def integrate(u, delta_t, n, c):
    a = 0.2
    b = 0.2
    du = lambda m: np.array((-m[1] - m[2], m[0] + a * m[1], b + m[2] * (m[0] - c)))     #First component is x, second component is y, third component is z
    @nb.jit
    def rk4_step(u_prev, delta_t, t_current):
        K1 = delta_t * du(u_prev)
        K2 = delta_t * du(u_prev + K1 / 2)
        K3 = delta_t * du(u_prev + K2 / 2)
        K4 = delta_t * du(u_prev + K3)# 4 intermediate approximations
        return u_prev + (K1 + 2 * K2 + 2 * K3 + K4) / 6
    @nb.jit
    def for_loop(u, delta_t, n):
        for i in range(n):
            t_current = delta_t * i
            u[i] = rk4_step(u[i - 1], delta_t, t_current)
        return u
    return for_loop(u, delta_t, n)

@nb.jit(nopython=True)
def rk4_step(u_prev, delta_t, t_current):
    du = lambda m: np.array((-m[1] - m[2], m[0] + a * m[1], b + m[2] * (m[0] - c)))     #First component is x, second component is y, third component is z
    K1 = delta_t * du(u_prev)
    K2 = delta_t * du(u_prev + K1 / 2)
    K3 = delta_t * du(u_prev + K2 / 2)
    K4 = delta_t * du(u_prev + K3)# 4 intermediate approximations
    return u_prev + (K1 + 2 * K2 + 2 * K3 + K4) / 6

def plotxyz_separate(sol):
    plots = plt.figure(1, figsize=(12,4))
    plx = plots.add_subplot(1,3,1)
    plotx(sol, plx)
    ply = plots.add_subplot(1,3,2)
    ploty(sol, ply)
    plz = plots.add_subplot(1,3,3)
    plotz(sol, plz)

def plotx(sol, plotx):
    plotx.set_title("$X$ versus $t$ Plot")
    plotx.set_xlabel("time ($s$)")
    plotx.set_ylabel("position ($m$)")
    plotx.set_ylim((-12,12))
    plotx.plot(sol['t'], sol['x'], 'k-', lw=0.5)

def ploty(sol, ploty):
    ploty.set_title("$Y$ versus $t$ Plot")
    ploty.set_xlabel("time ($s$)")
    ploty.set_ylabel("position ($m$)")
    ploty.set_ylim((-12,12))
    ploty.plot(sol['t'], sol['y'], 'k-', lw=0.5)

def plotz(sol, plotz):
    plotz.set_title("$Z$ versus $t$ Plot")
    plotz.set_xlabel("time ($s$)")
    plotz.set_ylabel("position ($m$)")
    plotz.set_ylim((0,25))
    plotz.plot(sol['t'], sol['z'], 'k-', lw=0.5)

def plot_2D_phase(sol, T0=100):
    sol_slice = sol[sol['t'] >= T0]
    plots = plt.figure(2, figsize=(12,5))
    plxy = plots.add_subplot(1,3,1)
    plotxy(sol_slice, T0, plxy)

def plotxy(sol, T0, fig):
    fig.set_title("$xy$ Parametric Plot")
    fig.set_xlabel("$x$ Position ($m$)")
    fig.set_ylabel("$y$ Position ($m$)")
    fig.plot((sol['x'], sol['y']))

def test_rossler():
    print(rossler(1, T=5)['x'])
    assert False
