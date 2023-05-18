from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

def simulate_model():
    # define the values of the model
    alpha = 0.5
    eta = 0.8
    mu = 0.1
    L_0 = 100
    L_t = L_0
    A = 2
    X = 3

    # define L_t, L^*, y_t and y^* for the simulations
    t_values = [0]
    L_t_values = [L_t]
    L_star_values = [((eta / mu) ** (1/alpha)) * A * X]
    y_t_values = [(A*X/L_t)**alpha]
    y_star_values = [mu/eta]


    # define simulation time parametres
    T = 30
    dt = 0.1

    # loop over time steps and calculate the equations of the model
    for t in range(1, int(T/dt) + 1):
        Y_t = L_t**(1-alpha)*(A*X)**alpha
        y_t = (A*X/L_t)**alpha
        n_t = eta*y_t
        L_t1 = n_t*L_t + (1-mu)*L_t
        L_t = L_t1
        L_star = ((eta / mu) ** (1/alpha)) * A * X
        y_star = mu/eta 
        #append time, labor force values and output per worker values to lists
        t_values.append(t*dt)
        L_t_values.append(L_t)
        L_star_values.append(L_star)
        y_t_values.append(y_t)
        y_star_values.append(y_star)

    # repeat the process, but now including a positive shock in A
    L_t=L_0
    A=2
    t_values_shock = [0]
    L_t_values_shock = [L_t]
    L_star_values_shock = [((eta / mu) ** (1/alpha)) * A * X]
    y_t_values_shock = [(A*X/L_t)**alpha]
    y_star_values_shock = [mu/eta]

    for t in range(1, int(T/dt) + 1):
        Y_t = L_t**(1-alpha)*(A*X)**alpha
        y_t = (A*X/L_t)**alpha
        n_t = eta*y_t
        if t == 100:
            A = 2.3 * A  # shock in parameter A
        L_t1 = n_t*L_t + (1-mu)*L_t
        L_t = L_t1
        L_star = ((eta / mu) ** (1/alpha)) * A * X
        t_values_shock.append(t*dt)
        L_t_values_shock.append(L_t)
        L_star_values_shock.append(L_star)
        y_t_values_shock.append(y_t)
        y_star_values_shock.append(y_star)

    # plot labor force simulation and steady state value
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(t_values, L_t_values, label='L_t')
    axs[0].plot(t_values, L_star_values, label='L_star')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Labor force')
    axs[0].set_title('Labour force simulation')
    axs[0].legend()

    # plot with a positive shock in A
    axs[1].plot(t_values_shock, L_t_values_shock, label='L_t')
    axs[1].plot(t_values_shock, L_star_values_shock, label='L_star')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Labor force')
    axs[1].set_title('Labor force with technology shock in period 10')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_y_t():
    # define the values of the model
    alpha = 0.5
    eta = 0.8
    mu = 0.1
    L_0 = 100
    L_t = L_0
    A = 2
    X = 3

     # define L_t, L^*, y_t and y^* for the simulations
    t_values = [0]
    L_t_values = [L_t]
    L_star_values = [((eta / mu) ** (1/alpha)) * A * X]
    y_t_values = [(A*X/L_t)**alpha]
    y_star_values = [mu/eta]


    # define simulation time parametres
    T = 30
    dt = 0.1

    # loop over time steps and calculate the equations of the model
    for t in range(1, int(T/dt) + 1):
        Y_t = L_t**(1-alpha)*(A*X)**alpha
        y_t = (A*X/L_t)**alpha
        n_t = eta*y_t
        L_t1 = n_t*L_t + (1-mu)*L_t
        L_t = L_t1
        L_star = ((eta / mu) ** (1/alpha)) * A * X
        y_star = mu/eta 
        #append time, labor force values and output per worker values to lists
        t_values.append(t*dt)
        L_t_values.append(L_t)
        L_star_values.append(L_star)
        y_t_values.append(y_t)
        y_star_values.append(y_star)

    # repeat the process, but now including a positive shock in A
    L_t=L_0
    A=2
    t_values_shock = [0]
    L_t_values_shock = [L_t]
    L_star_values_shock = [((eta / mu) ** (1/alpha)) * A * X]
    y_t_values_shock = [(A*X/L_t)**alpha]
    y_star_values_shock = [mu/eta]

    for t in range(1, int(T/dt) + 1):
        Y_t = L_t**(1-alpha)*(A*X)**alpha
        y_t = (A*X/L_t)**alpha
        n_t = eta*y_t
        if t == 100:
            A = 2.3 * A  # shock in parameter A
        L_t1 = n_t*L_t + (1-mu)*L_t
        L_t = L_t1
        L_star = ((eta / mu) ** (1/alpha)) * A * X
        t_values_shock.append(t*dt)
        L_t_values_shock.append(L_t)
        L_star_values_shock.append(L_star)
        y_t_values_shock.append(y_t)
        y_star_values_shock.append(y_star)

    # plot output per worker simulation and state state value
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(t_values, y_t_values, label='y_t')
    axs[0].plot(t_values, y_star_values, label='y_star')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Output per worker')
    axs[0].set_title('Output per worker simulation')
    axs[0].legend()

    # plot output per worker simulation and state state value with a positive shock to the technology curve
    axs[1].plot(t_values_shock, y_t_values_shock, label='y_t')
    axs[1].plot(t_values_shock, y_star_values_shock, label='y_star')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Output per worker')
    axs[1].set_title('Output per worker with technology shock in period 10')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def further_analysis():
    # define the values of the model
    alpha = 0.5
    eta = 0.8
    mu = 0.04
    L_0 = 100
    L_t = L_0
    A = 2
    X = 3

    # define L_t, L^*, y_t and y^* for the simulations
    t_values = [0]
    L_t_values = [L_t]
    L_star_values = [((eta / mu) ** (1/alpha)) * A * X]
    y_t_values = [(A*X/L_t)**alpha]
    y_star_values = [mu/eta]


    # define simulation time parametres
    T = 30
    dt = 0.1

    # loop over time steps and calculate the equations of the model
    for t in range(1, int(T/dt) + 1):
        Y_t = L_t**(1-alpha)*(A*X)**alpha
        y_t = (A*X/L_t)**alpha
        n_t = eta*y_t
        L_t1 = n_t*L_t + (1-mu)*L_t
        L_t = L_t1
        L_star = ((eta / mu) ** (1/alpha)) * A * X
        y_star = mu/eta 
        #append time, labor force values and output per worker values to lists
        t_values.append(t*dt)
        L_t_values.append(L_t)
        L_star_values.append(L_star)
        y_t_values.append(y_t)
        y_star_values.append(y_star)

    # repeat the process, but now including a positive shock in A
    L_t=L_0
    A=2
    t_values_shock = [0]
    L_t_values_shock = [L_t]
    L_star_values_shock = [((eta / mu) ** (1/alpha)) * A * X]
    y_t_values_shock = [(A*X/L_t)**alpha]
    y_star_values_shock = [mu/eta]

    for t in range(1, int(T/dt) + 1):
        Y_t = L_t**(1-alpha)*(A*X)**alpha
        y_t = (A*X/L_t)**alpha
        n_t = eta*y_t
        if t == 100:
            A = 2.3 * A  # shock in parameter A
        L_t1 = n_t*L_t + (1-mu)*L_t
        L_t = L_t1
        L_star = ((eta / mu) ** (1/alpha)) * A * X
        t_values_shock.append(t*dt)
        L_t_values_shock.append(L_t)
        L_star_values_shock.append(L_star)
        y_t_values_shock.append(y_t)
        y_star_values_shock.append(y_star)

    # plot labor force simulation and steady state value
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(t_values, L_t_values, label='L_t')
    axs[0].plot(t_values, L_star_values, label='L_star')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Labor force')
    axs[0].set_title('Labour force simulation')
    axs[0].legend()

    # plot with a positive shock in A
    axs[1].plot(t_values_shock, L_t_values_shock, label='L_t')
    axs[1].plot(t_values_shock, L_star_values_shock, label='L_star')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Labor force')
    axs[1].set_title('Labor force with technology shock in period 10')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def further_analysis_y_t():
        # define the values of the model
    alpha = 0.5
    eta = 0.8
    mu = 0.04
    L_0 = 100
    L_t = L_0
    A = 2
    X = 3

    # define L_t, L^*, y_t and y^* for the simulations
    t_values = [0]
    L_t_values = [L_t]
    L_star_values = [((eta / mu) ** (1/alpha)) * A * X]
    y_t_values = [(A*X/L_t)**alpha]
    y_star_values = [mu/eta]


    # define simulation time parametres
    T = 30
    dt = 0.1

    # loop over time steps and calculate the equations of the model
    for t in range(1, int(T/dt) + 1):
        Y_t = L_t**(1-alpha)*(A*X)**alpha
        y_t = (A*X/L_t)**alpha
        n_t = eta*y_t
        L_t1 = n_t*L_t + (1-mu)*L_t
        L_t = L_t1
        L_star = ((eta / mu) ** (1/alpha)) * A * X
        y_star = mu/eta 
        #append time, labor force values and output per worker values to lists
        t_values.append(t*dt)
        L_t_values.append(L_t)
        L_star_values.append(L_star)
        y_t_values.append(y_t)
        y_star_values.append(y_star)

    # repeat the process, but now including a positive shock in A
    L_t=L_0
    A=2
    t_values_shock = [0]
    L_t_values_shock = [L_t]
    L_star_values_shock = [((eta / mu) ** (1/alpha)) * A * X]
    y_t_values_shock = [(A*X/L_t)**alpha]
    y_star_values_shock = [mu/eta]

    for t in range(1, int(T/dt) + 1):
        Y_t = L_t**(1-alpha)*(A*X)**alpha
        y_t = (A*X/L_t)**alpha
        n_t = eta*y_t
        if t == 100:
            A = 2.3 * A  # shock in parameter A
        L_t1 = n_t*L_t + (1-mu)*L_t
        L_t = L_t1
        L_star = ((eta / mu) ** (1/alpha)) * A * X
        t_values_shock.append(t*dt)
        L_t_values_shock.append(L_t)
        L_star_values_shock.append(L_star)
        y_t_values_shock.append(y_t)
        y_star_values_shock.append(y_star)
    # plot output per worker simulation and state state value
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(t_values, y_t_values, label='y_t')
    axs[0].plot(t_values, y_star_values, label='y_star')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Output per worker')
    axs[0].set_title('Output per worker simulation')
    axs[0].legend()

    # plot output per worker simulation and state state value with a positive shock to the technology curve
    axs[1].plot(t_values_shock, y_t_values_shock, label='y_t')
    axs[1].plot(t_values_shock, y_star_values_shock, label='y_star')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Output per worker')
    axs[1].set_title('Output per worker with technology shock in period 10')
    axs[1].legend()

    plt.tight_layout()
    plt.show()