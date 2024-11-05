import numpy as np
from pricing_engine import QMC_fourier_pricing_engine

def main():
    model_name = 'GH'
    payoff_name = 'put_on_max'  # Change this to test different payoffs
    
    # Define parameters
    d = 6  # Example dimension
    S0 = 100 * np.ones(d)  # Initial stock prices
    K = 100  # Strike price
    r = 0.1 # Risk-free rate
    T = 1.0   # Time to maturity
    q = 0.05     # Dividend yield
    alpha = 10
    beta = -3 * np.ones(d)
    delta = 0.2
    DELTA = np.identity(d)
    lambda_var = -1

    # Weights for the basket option, provided by the user or another source
    weights = 1/d * np.ones(d)

    option_params = [d, S0, K, r, T, q, alpha, beta, delta, DELTA, lambda_var]


    N_samples = 2**10
    S_shifts = 30
    
    # Optionally specify transform distribution and parameters if needed
    transform_distribution = None  # or 'ML', 'MT', 'MN' based on your choice
    transform_params = None  # Set if specifying a custom transform distribution

    if payoff_name == 'basket_put':
        price_estimate, error_estimate = QMC_fourier_pricing_engine(
            model_name,
            payoff_name,
            option_params,
            N_samples,
            S_shifts,
            transform_distribution=transform_distribution,
            transform_params=transform_params,
            weights=weights  # Pass weights only for basket put
        )
    else:
        price_estimate, error_estimate = QMC_fourier_pricing_engine(
            model_name,
            payoff_name,
            option_params,
            N_samples,
            S_shifts,
            transform_distribution=transform_distribution,
            transform_params=transform_params
        )

    print(f"Estimated Price: {round(price_estimate,5)}, Statistical Error: {round(error_estimate,5)}, Relative Error: {round(error_estimate / price_estimate,5)}")

if __name__ == "__main__":
    main()