# damping_optimization.py
import numpy as np
import scipy.optimize
from payoffs import CallOnMin, PutOnMax, SpreadCall, BasketPut, CashOrNothingPut  # Import the payoff classes
import warnings
warnings.simplefilter('ignore')

def optimize_damping_parameters(payoff, model, d, S0, K):
    """
    Optimizes the damping parameters R based on the given model and payoff.
    
    Parameters:
    - payoff: The payoff object that contains methods for calculating X0 and constraints.
    - model: The model object that contains methods for characteristic function and constraints.
    - d: Number of Assets.
    - S0: Initial stock prices.
    - K: Strike price.
    
    Returns:
    - optimal_R: The optimized damping parameters.
    """
    
    # Calculate X0 based on payoff and initial conditions
    X0 = payoff.calculate_X0(S0, K)
    
    # Define constraints from both model and payoff
    cons = []
    cons.append({'type': 'ineq', 'fun': model.strip_of_analyticity()})
    cons.extend(payoff.strip_of_analyticity())
    
    # Define the function to optimize (log of absolute value of integrand)
    def log_integrand_to_optimize(R):
        return np.log(np.abs(np.exp(-R @ X0) * model.characteristic_function(1j*R) * payoff.fourier_transform(1j*R)))
    
    # Conditionally define R_init by a feasible point in the strip of analyticity of the integrand based on the type of payoff
    if isinstance(payoff, CallOnMin):
        R_init = -2 * np.ones(d)  # Custom initial value for CallOnMin
    elif isinstance(payoff, PutOnMax):
        R_init = 2 * np.ones(d)   # Custom initial value for PutOnMax
    elif isinstance(payoff, SpreadCall):
        R_init = np.ones(d)
        R_init[0] = -1 - d        # Custom initial value for SpreadCall
    elif isinstance(payoff, BasketPut):
        R_init = 2 * np.ones(d)   # Custom initial value for BasketPut
    elif isinstance(payoff, CashOrNothingPut):
        R_init = 2 * np.ones(d)   # Custom initial value for BasketPut
    else:
        raise ValueError(f"Unsupported payoff type: {type(payoff)}")
    
    print("R_init = ", R_init)
    
    # Perform optimization using scipy.optimize.minimize
    optimal_R = scipy.optimize.minimize(
        fun=log_integrand_to_optimize,
        constraints=cons,
        x0=R_init,
        method="Trust-Constr"
    )
    
    return optimal_R.x