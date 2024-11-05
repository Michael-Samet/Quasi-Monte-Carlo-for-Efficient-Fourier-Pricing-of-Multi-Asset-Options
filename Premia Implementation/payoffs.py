import numpy as np
from scipy.special import gamma 
import numpy as np

class CallOnMin:
    """Class representing a call option on the minimum of multiple assets.
    
    Methods:
        fourier_transform(u): Computes the Fourier transform for the option payoff.
        calculate_X0(S0, K): Defines X0 
        scaling(K): Returns a scaling factor based on the definition of X0
        strip_of_analyticity(): Defines the strip of analyticity of the payoff transform.
        initial_R(d): Provides an initial guess for `R` when finding the optimal damping parameters.
    """
    
    def fourier_transform(self, u: np.ndarray) -> complex:
        """Computes the Fourier transform of the payoff function.

        Args:
            u (np.ndarray): Complex argument for which to evaluate the Fourier transform.

        Returns:
            complex: The value of the Fourier transform at `u`.

        """
        
        # Compute denominator involving summation and product of `u`
        denominator = (1j * np.sum(u) - 1) * np.prod(1j * u)
        
        # Return reciprocal of the denominator as Fourier transform result
        return 1 / denominator
   
    def calculate_X0(self, S0: np.ndarray, K: float) -> np.ndarray:
        """Defines X0
        Args:
            S0 (np.ndarray): Spot prices of the underlying assets.
            K (float): Strike price of the option.

        Returns:
            np.ndarray: Logarithm of `S0 / K`.

        """
        
        return np.log(S0 / K)
    
    def scaling(self, K: float) -> float:
        """Returns a scaling factor based on the strike price.

        This method scales the integrand appropriately based on the definition of X0

        Args:
            K (float): Strike price of the option.

        Returns:
            float: The scaling factor

        """
        
        # Return strike price as scaling factor
        return K
    
    def strip_of_analyticity(self):
        """Defines constraints for the strip of analyticity.

        Returns:
            list[dict]: A list of inequality constraints for the optimization of damping parameters.
        
         """
         
         # Define two inequality constraints for strip of analyticity
        return [
             {'type': 'ineq', 'fun': lambda R: -R},  # Ensures R is non-positive
             {'type': 'ineq', 'fun': lambda R: -1 - np.sum(R)}  # Ensures sum(R) <= -1
         ]
        


class PutOnMax:
    """Class representing a put option on the maximum of multiple assets.
    
    Methods:
        fourier_transform(u): Computes the Fourier transform for the option payoff.
        calculate_X0(S0, K): Defines X0 based on spot prices and strike price.
        scaling(K): Returns a scaling factor based on the definition of X0.
        strip_of_analyticity(): Defines the strip of analyticity of the payoff transform.
        initial_R(d): Provides an initial guess for `R` when finding the optimal damping parameters.
    """
    
    def fourier_transform(self, u: np.ndarray) -> complex:
        """Computes the Fourier transform of the payoff function.

        Args:
            u (np.ndarray): Complex argument for which to evaluate the Fourier transform.

        Returns:
            complex: The value of the Fourier transform at `u`.

        """
        
        # Number of dimensions (assets)
        d = len(u)
        
        # Compute denominator involving summation and product of `u`
        denominator = (-1)**d * (1 - 1j * np.sum(u)) * np.prod(1j * u)
        
        # Return reciprocal of the denominator as Fourier transform result
        return 1 / denominator
   
    def calculate_X0(self, S0: np.ndarray, K: float) -> np.ndarray:
        """Defines X0 based on spot prices and strike price.

        Args:
            S0 (np.ndarray): Spot prices of the underlying assets.
            K (float): Strike price of the option.

        Returns:
            np.ndarray: Logarithm of `S0 / K`.

        """
        
        return np.log(S0 / K)
    
    def scaling(self, K: float) -> float:
        """Returns a scaling factor based on the strike price.

        This method scales the integrand appropriately based on the definition of X0.

        Args:
            K (float): Strike price of the option.

        Returns:
            float: The scaling factor.

        """
        
        # Return strike price as scaling factor
        return K
    
    def strip_of_analyticity(self):
        """Defines constraints for the strip of analyticity.

        Returns:
            list[dict]: A list of inequality constraints for the optimization of damping parameters.
        
         """
         
         # Define inequality constraint for strip of analyticity
        return [
             {'type': 'ineq', 'fun': lambda R: R},  # Ensures R is non-negative
         ]
        
   

class SpreadCall:
    """Class representing a spread call option on multiple assets.
    
    Methods:
        fourier_transform(u): Computes the Fourier transform for the spread call option payoff.
        calculate_X0(S0, K): Defines X0 based on spot prices and strike price.
        scaling(K): Returns a scaling factor based on the definition of X0.
        strip_of_analyticity(): Defines the strip of analyticity of the payoff transform.
        initial_R(d): Provides an initial guess for `R` when finding the optimal damping parameters.
    """
    
    def fourier_transform(self, u: np.ndarray) -> complex:
        """Computes the Fourier transform of the spread call option payoff.

        Args:
            u (np.ndarray): Complex argument for which to evaluate the Fourier transform.

        Returns:
            complex: The value of the Fourier transform at `u`.

        """
        
        # Compute numerator using gamma functions
        numerator = gamma(1j * (u[0] + np.sum(u[1:])) - 1) * np.prod(gamma(-1j * u[1:]))
        
        # Compute denominator using gamma function
        denominator = gamma(1j * u[0] + 1)
        
        # Return the ratio of numerator to denominator as Fourier transform result
        return numerator / denominator

    def calculate_X0(self, S0: np.ndarray, K: float) -> np.ndarray:
        """Defines X0 based on spot prices and strike price.

        Args:
            S0 (np.ndarray): Spot prices of the underlying assets.
            K (float): Strike price of the option.

        Returns:
            np.ndarray: Logarithm of `S0 / K`.

        """
        
        return np.log(S0 / K)

    def scaling(self, K: float) -> float:
        """Returns a scaling factor based on the strike price.

        This method scales the integrand appropriately based on the definition of X0.

        Args:
            K (float): Strike price of the option.

        Returns:
            float: The scaling factor.

        """
        
        # Return strike price as scaling factor
        return K

    def strip_of_analyticity(self):
        """Defines constraints for the strip of analyticity.

        This method defines two constraints: 
          - Components 2 to d must be positive.
          - The first component must satisfy a specific inequality.

        Returns:
            list[dict]: A list of inequality constraints for the optimization of damping parameters.
        
         """
         
         # Constraint for components 2 to d to be positive (R[1:] > 0)
        positivity_constraints = lambda R: R[1:]
         
         # Constraint for the first component (R[0])
        first_component_constraint = lambda R: -1 - np.sum(R[1:]) - R[0]
        
         # Combine all constraints into a list
        return [
             {'type': 'ineq', 'fun': positivity_constraints},
             {'type': 'ineq', 'fun': first_component_constraint}
         ]
    
 



class BasketPut:
    """Class representing a basket put option on multiple assets.
    
    Methods:
        fourier_transform(u): Computes the Fourier transform for the basket put option payoff.
        calculate_X0(S0, K): Defines X0 based on spot prices, strike price, and asset weights.
        scaling(K): Returns a scaling factor based on the definition of X0.
        strip_of_analyticity(): Defines the strip of analyticity of the payoff transform.
        initial_R(d): Provides an initial guess for `R` when finding the optimal damping parameters.
    """
    
    def __init__(self, weights: np.ndarray):
        """Initializes an instance of the BasketPut class.

        Args:
            weights (np.ndarray): Weights assigned to each asset in the basket.
        """
        self.weights = weights
        
    def fourier_transform(self, u: np.ndarray) -> complex:
        """Computes the Fourier transform of the payoff for a scaled (K = 1) basket put option.

        Args:
            u (np.ndarray): Array of Fourier frequencies.

        Returns:
            complex: The value of the Fourier transform at `u`.

        """
        
        # Compute numerator using gamma functions
        numerator = np.prod(gamma(-1j * u))
        
        # Compute denominator using gamma function
        denominator = gamma(-1j * np.sum(u) + 2)
        
        # Return the ratio of numerator to denominator as Fourier transform result
        return numerator / denominator
   
    def calculate_X0(self, S0: np.ndarray, K: float) -> np.ndarray:
        """Defines X0 based on spot prices, strike price, and asset weights.

        Args:
            S0 (np.ndarray): Spot prices of the underlying assets.
            K (float): Strike price of the option.

        Returns:
            np.ndarray: Logarithm of weighted `S0 / K`.

        """
        
        return np.log(self.weights * S0 / K)

    def scaling(self, K: float) -> float:
        """Returns a scaling factor based on the strike price.

        This method scales the integrand appropriately based on the definition of X0.

        Args:
            K (float): Strike price of the option.

        Returns:
            float: The scaling factor.

        """
        
        # Return strike price as scaling factor
        return K

    def strip_of_analyticity(self):
        """Defines constraints for the strip of analyticity.

        This method defines a constraint ensuring that R is non-negative.

        Returns:
            list[dict]: A list of inequality constraints for the optimization of damping parameters.
        
         """
         
         # Define inequality constraint for strip of analyticity (R > 0)
        return [
             {'type': 'ineq', 'fun': lambda R: R},  # Ensures R is non-negative
         ]
    
    
class CashOrNothingPut:
    """Class representing a cash or nothing put option on multiple assets.
    
    Methods:
        fourier_transform(u): Computes the Fourier transform for the basket put option payoff.
        calculate_X0(S0, K): Defines X0 based on spot prices, strike price, and asset weights.
        scaling(K): Returns a scaling factor based on the definition of X0.
        strip_of_analyticity(): Defines the strip of analyticity of the payoff transform.
        initial_R(d): Provides an initial guess for `R` when finding the optimal damping parameters.
    """
        
    def fourier_transform(self, u: np.ndarray) -> complex:
        """Computes the Fourier transform of the payoff for a scaled (K = 1) basket put option.

        Args:
            u (np.ndarray): Array of Fourier frequencies.
            K (float): Strike price of the option.

        Returns:
            complex: The value of the Fourier transform at `u`.

        """
        
        # Compute numerator using gamma functions
        numerator = -1 
        
        # Compute denominator using gamma function
        denominator =1j * u
        
        # Return the ratio of numerator to denominator as Fourier transform result
        return np.prod(numerator / denominator)
   
    def calculate_X0(self, S0: np.ndarray, K: float) -> np.ndarray:
        """Defines X0 based on spot prices, strike price, and asset weights.

        Args:
            S0 (np.ndarray): Spot prices of the underlying assets.
            K (float): Strike price of the option.

        Returns:
            np.ndarray: Logarithm of weighted `S0 / K`.

        """
        
        return np.log(S0 / K)

    def scaling(self, K: float) -> float:
        """Returns a scaling factor based on the strike price.

        This method scales the integrand appropriately based on the definition of X0.

        Args:
            K (float): Strike price of the option.

        Returns:
            float: The scaling factor.

        """
        
        # Return strike price as scaling factor
        return 1

    def strip_of_analyticity(self):
        """Defines constraints for the strip of analyticity.

        This method defines a constraint ensuring that R is non-negative.

        Returns:
            list[dict]: A list of inequality constraints for the optimization of damping parameters.
        
         """
         
         # Define inequality constraint for strip of analyticity (R > 0)
        return [
             {'type': 'ineq', 'fun': lambda R: R},  # Ensures R is non-negative
         ]
    
   
   