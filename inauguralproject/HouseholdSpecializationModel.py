from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 
        par.phi = 1

        # household production
        par.alpha = 0.5
        par.sigma = 1.0

        # wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # consumption of market goods
        C = par.wM*LM + par.phi*par.wF*LF

        # home production
        if par.sigma == 0:
            H = min(HM, HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**((par.sigma)/(par.sigma-1))

        # total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF 
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    
    def solve(self,do_print=False):
        """ solve model continously """
        
        opt = SimpleNamespace()

        # Define the objective function
        def objective(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF)

        # Define constraints and bounds
        def constraints(x):
            LM, HM, LF, HF = x
            return [24 - LM - HM, 24 - LF - HF]
        
        constraints = ({'type':'ineq', 'fun': constraints})
        bounds = ((0,24),(0,24),(0,24),(0,24))

        # Initial guess
        initial_guess = [6, 6, 6, 6]

        # Call the solver
        solution = optimize.minimize(
            objective, initial_guess, 
            method='Nelder-Mead', 
            bounds=bounds, 
            constraints = constraints
            )
        
        opt.LM, opt.HM, opt.LF, opt.HF = solution.x

        return opt
   

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        # Fill out solution vectors for HF and HM
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF
            optimum = self.solve()
            sol.HF_vec[i] = optimum.HF
            sol.HM_vec[i] = optimum.HM
            sol.LF_vec[i] = optimum.LF
            sol.LM_vec[i] = optimum.LM
        
        return sol.HF_vec, sol.HM_vec, sol.LF_vec, sol.LM_vec

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]


    def estimate(self,alpha=None, phi=None, do_print = False):
        """ estimate alpha and sigma for variable and fixed alpha """

        # set par and sol 
        par = self.par
        sol = self.sol

        # if alpha is not given  
        if phi == None:
            # objective function (to minimize) 
            def objective(y):
                par.phi = y[1]
                # varibale sigma 
                par.sigma = y[0] 
                par.alpha = alpha
                # solve solve_wF_vec
                self.solve_wF_vec()
                # run regression
                self.run_regression()
                # return the diffrence between target beta and estimated 
                return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2
            # define objective function 
            obj = lambda y: objective(y)
            # guess for alpha and sigma  
            guess = [0.5]*2
            # bounds
            bounds = [(-0.00001,1)]*2
            # optimizer
            result = optimize.minimize(obj,
                                guess,
                                method='Nelder-Mead',
                                bounds=bounds)
            
            # print result 
            if do_print:
                print(f'alpha = {result.x[1].round(4)}')
                print(f'sigma = {result.x[0].round(4)}')
                print(f'phi = {result.x[2].round(4)}')
        
        # if alpha and phi is given  
        else:
            # objective function (to minimize)
            def objective(y):
                # chosen alpha
                par.alpha = alpha 
                # variables
                par.sigma = y[0] 
                # phi
                par.phi = phi
                # solve solve_wF_vec
                self.solve_wF_vec()
                # run regression 
                self.run_regression()
                # return the diffrence between target beta and estimated 
                return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2
            # define objective function 
            obj = lambda y: objective(y)
            # guess for sigma 
            guess = [0.5]
            # bounds 
            bounds = [(-0.00001,1)]
            # optimizer
            result = optimize.minimize(obj,
                                guess,
                                method = 'Nelder-Mead',
                                bounds = bounds)
            # print result
            if do_print: 
                print(f'sigma = {result.x[0].round(4)}')

        return result