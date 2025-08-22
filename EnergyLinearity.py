import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Energy linearity plots

energy = np.array([662, 835, 1275, 1332]) # Known energies in keV for Cs137, Mn54, Na22, and Co60
ADC_counts = np.array([5511.95, 11553.41, 14370.81, 14778.19]) # Corresponding ADC counts from the fits
errors = np.array([24.49, 45.99, 112.65, 62.92]) # Errors in ADC counts from the fits

# Linear fit function
def linear(x, m, b):
    return m * x + b

# Perform the linear fit

par, cov = optimize.curve_fit(linear, energy, ADC_counts, sigma=errors)
m_fit, b_fit = np.round(par, decimals=2)
m_err, b_err = np.round(np.sqrt(np.diag(cov)), decimals=2)
print(f"Fitted parameters: m = {m_fit} ± {m_err}, b = {b_fit} ± {b_err}")

# Plotting the energy linearity

plt.errorbar(energy, ADC_counts, yerr=errors, fmt='o', label='Data')
x_fit = np.linspace(0, 1500, 100)
y_fit = linear(x_fit, m_fit, b_fit)
plt.plot(x_fit, y_fit, 'r--', label=f'Fit: y = {m_fit}x + {b_fit}')
plt.xlabel('Energy (keV)')
plt.ylabel('ADC Counts')
plt.title('Energy Linearity for Gain 10000 GUI Setting')
plt.legend()
plt.grid()
plt.show()