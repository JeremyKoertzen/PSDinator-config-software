import numpy as np
import matplotlib.pyplot as plt
from test_unpacker_for_v031_heatmap import gaussFit

# Gamma source data log plot

# Importing the data from text files

Na22 = np.loadtxt("Na22.txt", skiprows=1)
Na22_bin_centers = Na22[:, 0]
Na22_counts = Na22[:, 1]
Na22_bin_size = Na22_bin_centers[1] - Na22_bin_centers[0]

Cs137 = np.loadtxt("Cs137.txt", skiprows=1)
Cs137_bin_centers = Cs137[:, 0]
Cs137_counts = Cs137[:, 1]
Cs137_bin_size = Cs137_bin_centers[1] - Cs137_bin_centers[0]

Mn54 = np.loadtxt("Mn54.txt", skiprows=1)
Mn54_bin_centers = Mn54[:, 0]
Mn54_counts = Mn54[:, 1]
Mn54_bin_size = Mn54_bin_centers[1] - Mn54_bin_centers[0]

Co60 = np.loadtxt("Co60.txt", skiprows=1)
Co60_bin_centers = Co60[:, 0]
Co60_counts = Co60[:, 1]
Co60_bin_size = Co60_bin_centers[1] - Co60_bin_centers[0]

# Calling the function to do the gaussian fit

Cs137_left = 4900
Cs137_right = 6000
Cs137_x, Cs137_y_fit, Cs137_x_compton, Cs137_y_compton, Cs137_compton_edge, Cs137_x0_err = gaussFit(Cs137_bin_centers, Cs137_counts, Cs137_bin_size, Cs137_left, Cs137_right, "Cs137")

Mn54_left = 10600
Mn54_right = 12500
Mn54_x, Mn54_y_fit, Mn54_x_compton, Mn54_y_compton, Mn54_compton_edge, Mn54_x0_err = gaussFit(Mn54_bin_centers, Mn54_counts, Mn54_bin_size, Mn54_left, Mn54_right, "Mn54")

Na22_left = 14070
Na22_right = 15000
Na22_x, Na22_y_fit, Na22_x_compton, Na22_y_compton, Na22_compton_edge, Na22_x0_err = gaussFit(Na22_bin_centers, Na22_counts, Na22_bin_size, Na22_left, Na22_right, "Na22")

Co60_left = 14430
Co60_right = 15160
Co60_x, Co60_y_fit, Co60_x_compton, Co60_y_compton, Co60_compton_edge, Co60_x0_err = gaussFit(Co60_bin_centers, Co60_counts, Co60_bin_size, Co60_left, Co60_right, "Co60")

# Plotting the data

fig, (Cs137, Mn54, Na22, Co60) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

Cs137.plot(Cs137_bin_centers, Cs137_counts, label='Cs137', color='orange')
Cs137.plot(Cs137_x, Cs137_y_fit, 'r--', label='Gaussian Fit')
Cs137.plot(Cs137_x_compton, Cs137_y_compton, 'g-', label='Compton Edge')
Cs137.set_yscale('log')
Cs137.set_ylabel('Counts')
Cs137.set_title('Cs137 Source Data (Log Scale)')
Cs137.legend()
Cs137.grid()

Mn54.plot(Mn54_bin_centers, Mn54_counts, label='Mn54', color='green')
Mn54.plot(Mn54_x, Mn54_y_fit, 'r--', label='Gaussian Fit')
Mn54.plot(Mn54_x_compton, Mn54_y_compton, 'g-', label='Compton Edge')
Mn54.set_yscale('log')
Mn54.set_ylabel('Counts')
Mn54.set_title('Mn54 Source Data (Log Scale)')
Mn54.legend()
Mn54.grid()

Na22.plot(Na22_bin_centers, Na22_counts, label='Na22', color='blue')
Na22.plot(Na22_x, Na22_y_fit, 'r--', label='Gaussian Fit')
Na22.plot(Na22_x_compton, Na22_y_compton, 'g-', label='Compton Edge')
Na22.set_yscale('log')
Na22.set_ylabel('Counts')
Na22.set_title('Na22 Source Data (Log Scale)')
Na22.legend()
Na22.grid()

Co60.plot(Co60_bin_centers, Co60_counts, label='Co60', color='black')
Co60.plot(Co60_x, Co60_y_fit, 'r--', label='Gaussian Fit')
Co60.plot(Co60_x_compton, Co60_y_compton, 'g-', label='Compton Edge')
Co60.set_yscale('log')
Co60.set_xlabel('ADC Counts')
Co60.set_ylabel('Counts')
Co60.set_title('Co60 Source Data (Log Scale)')
Co60.legend()
Co60.grid()

plt.show()
