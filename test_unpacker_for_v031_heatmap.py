#!/usr/bin/env python3
import argparse

# *************************************************************
# Program to get adc data from FPGA over USB port
#
# Author:   G. Engel
# Date:     7-Nov-2024
# Comment:  Handles receipt of timestamp counter
#           in form we expect for actual experiment
# Modifid on 21-Mar-2025
# Modified again on 15-Apr-2025 to add plotting of timestamps
# Modified on Jul 14 - Prince
# Modified on Jul 22 - Prince
# Byte #
#   0       Board ID (0 - 63)
#   1-4     Event number
#   5       Channel number
#   6       ADC A (low)
#   7       ADC A (high)
#   8       ADC B (low)
#   9       ADC B (high)
#   10      ADC C (low)
#   11      ADC C (high)
#   12      ADC T (low)
#   13      ADC T (high)
#
# and so on ...
#
# OR
#
#   0       Board ID = 255
#   1-4     Event number
#   5-11    Timestamp
# *************************************************************

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Get the libraries we need
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Need the numpy library

import numpy as np

# Need ctypes library since COBS routine is written in C

from ctypes import *
import ctypes

# Need the serial library

import serial

# Importing the required module for plotting

import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize

# So we can get environment variable info

import os

#So we can process multiple .bin files simultaneously

import glob
import re
import contextlib

#For importing .json files

import json


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Create best fit line
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def best_fit_line(x, slope, intercept):
    yFit = []
    for i in range(0, len(x)):
        yFit.append(slope * x[i] + intercept)
    return x, yFit


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Create the residuals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def residuals(x, y, yBestFit):
    yres = []
    for i in range(0, len(x)):
        yres.append(1e12 * (y[i] - yBestFit[i]))
    return x, yres


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Routine to plot timestamp data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def plot_event_times():
    # Set the figure size (16.8 inches wide and 7.2 inches high) 1680 x 720 is resolution

    plt.rcParams.update({'figure.figsize': (16.8, 7.2), 'figure.dpi': 100})

    # Perform linear regression analysis
    # Remove first data point before calling linregress()

    x = xData[1:]
    y = yData[1:]
    res = stats.linregress(x, y)

    # Plot original data (2 rows, 1 column)

    fig, axes = plt.subplots(2, 1)

    # Plot the best fit line

    xfit, yfit = best_fit_line(x, res.slope, res.intercept)
    axes[EVENT_TIMES].plot(xfit, yfit, color='red')

    # Plot the residuals
    xRes, yRes = residuals(x, y, yfit)
    axes[RESIDUALS].plot(xRes, yRes, color='blue')

    # Axes and titles

    axes[EVENT_TIMES].set_title("Event Times as Function of Event Number")
    axes[EVENT_TIMES].set_xlabel("Event Number")
    axes[EVENT_TIMES].set_ylabel("Event Times (sec)")

    axes[RESIDUALS].set_title("Residual as Function of Event Number")
    axes[RESIDUALS].set_xlabel("Event Number")
    axes[RESIDUALS].set_ylabel("Residual (ps)")

    # Add a grid to each of the plots

    axes[EVENT_TIMES].grid()
    axes[RESIDUALS].grid()

    slope_us = 1e6 * res.slope
    r_val = res.rvalue
    string = "Slope = {0:5.1f} usec per event with correlation coeff of {1:5.2f}".format(slope_us, r_val)
    axes[EVENT_TIMES].annotate(string, xy=(0.5, 0.9), xycoords='figure fraction')

    # Display the plot
    # Use tight layout to help make it look pretty

    fig.tight_layout()
    plt.show()


# Save as a PDF

#    plotFilename = "event_times.pdf"
#    plt.savefig(plotFilename)
#    plt.close()


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Routine to take two bytes and convert to
# ADC value (2's complement)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def bytes_to_adc(lower_byte, upper_byte, tag):
    adc_value = (upper_byte << 8) | lower_byte
    if adc_value >= (1 << 15):
        adc_value = adc_value - (1 << 16)
    print(f"......... {tag} -> {adc_value}")
    return


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Routine to take TIME1, TIME2, CALIBRATION1
# and CALIBRATION2 values and convert to a
# fraction of a period.
# ADC value (2's complement)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def tdc7200_calc(time1, time2, calib1, calib2):
    refClkFreq = 10e6
    refClkPeriod = 1 / refClkFreq
    calCount = calib2 - calib1
    convert_to_ns = refClkPeriod / 1e-9
    tof1 = time1 / calCount
    tof2 = time2 / calCount
    tof1 = convert_to_ns * tof1
    tof2 = convert_to_ns * tof2
    diff_tof = tof2 - tof1
    print(f"......... TOF1 -> {tof1:8.3f} ns")
    print(f"......... TOF2 -> {tof2:8.3f} ns")
    print(f"......... (TOF2 - TOF1) -> {diff_tof:8.3f} ns")
    return


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Routine to handle a packet from board 255 (0xff)
# which is a special timestamp packet!
# Fixed length packet (12 bytes)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def process_special_packet(packet_len, packet, event_number):
    # If this routine is called then board id is 255 (0xff)
    # Get timestamp counter

    tstamp = 0
    for k in range(0, 7, 1):
        byte = packet[5 + k]
        tstamp = tstamp | (byte << (8 * k))

    tstamp = tstamp / 4096.0
    event_time = 100e-9 * tstamp
    print(f"... Timestamp  -> {tstamp}")
    print(f"... Event time -> {event_time:15.12f} sec")

    # Append values to our plot arrays

    xData.append(event_number)
    yData.append(event_time)

    # We can return

    return


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Routine to handle an ordinary packet
# Variable length packet -> 5 + 9 * channel_count
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def process_ordinary_packet(packet_len, packet, event_number):
    # Extract the ADC data (9 bytes to a channel)

    for k in range(5, packet_len, 9):
        addr = packet[k]
        print(f"...... Channel -> {addr}")
        bytes_to_adc(packet[k + 1], packet[k + 2], 'A')
        bytes_to_adc(packet[k + 3], packet[k + 4], 'B')
        bytes_to_adc(packet[k + 5], packet[k + 6], 'C')
        bytes_to_adc(packet[k + 7], packet[k + 8], 'T')
    return


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Routine to process an event
# and send to print_data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def process_event(packet_len, packet, packet_counter):
    # Extract the event number (32-bit)

    event_number = packet[1]
    event_number += packet[2] << 8
    event_number += packet[3] << 16
    event_number += packet[4] << 24

    # Print out header line

    print(f"Packet #{packet_counter}, Event #{event_number} ")

    # Extract board id

    board_id = packet[0]
    print(f"... Board -> {board_id}")

    # If board is 255 then we have a special packet else
    # we have an ordinary packet

    if (board_id == 255):
        process_special_packet(packet_len, packet, event_number)
    else:
        process_ordinary_packet(packet_len, packet, event_number)
    return


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Routine to read in binary data and recover the original event data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def get_event_packets(fid_bin):
    # Create my_c_functions object

    so_file = "./cobs.so"
    COBS = CDLL(so_file)

    # Open up file to read binary data
    # Store into an uint8 array

    data = np.fromfile(fid_bin, dtype=np.uint8)

    # Convert numPy array to c type array

    c_data = (ctypes.c_uint8 * len(data))(*data)

    # Create maximum sized cobs and event packet arrays

    cobs_packet = (ctypes.c_uint8 * 255)()
    event_packet = (ctypes.c_uint8 * 253)()
    len_c_data = len(c_data)
    print(f"Length of cobs encode data file = {len_c_data} bytes")

    # Go through the data looking for NUL terminated event records
    # For each event, print out the event data

    packet_counter = 1
    cobs_packet_len = 0
    for i in range(len_c_data):
        byte = c_data[i]
        cobs_packet[cobs_packet_len] = byte
        cobs_packet_len = cobs_packet_len + 1
        if (byte == 0):
            event_packet_len = COBS.cobsDecode(cobs_packet, cobs_packet_len, event_packet)
            process_event(event_packet_len, event_packet, packet_counter)
            packet_counter += 1
            cobs_packet_len = 0
    return

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# JSON file reader
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def jsonReader(json_file):

    # Open and load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Extracting the values from the JSON data
    delay_a = np.array(data['configuration']['0']['psd']['octal_dac_settings']['delay_voltages']['a'])
    delay_b = np.array(data['configuration']['0']['psd']['octal_dac_settings']['delay_voltages']['b'])
    delay_c = np.array(data['configuration']['0']['psd']['octal_dac_settings']['delay_voltages']['c'])
    delay_input = np.array(data['configuration']['0']['delay']['1']['value'])

    #print("Delay values from JSON file:", delay)

    #return(delay_a, delay_b, delay_c)
    return(delay_input)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Histogram Binning Utility
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def histogramBinning(c_array):
    
    # Making bins and edges for the histogram

    bins, edges = np.histogram(c_array, bins=1000)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_size = edges[1] - edges[0]

    print(f"Leftmost bin center: {bin_centers[0]}, Rightmost bin center: {bin_centers[-1]}")

    # Saving the histogram data to a text file

    histogram_data = np.column_stack((bin_centers, bins))
    np.savetxt("histogram_data.txt", histogram_data, fmt="%.6f", header="Bin_Centers Counts")

    # Returning the bin centers and counts for further use

    return bin_centers, bins, bin_size

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Gauss Fit Utility
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def gaussFit(bin_centers, bins, bin_size, left, right, source):
    # Defining the Gaussian function for fitting

    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    # Saving the histogram data to a text file

    histogram_data = np.column_stack((bin_centers, bins))
    np.savetxt(f"{source}.txt", histogram_data, fmt="%.6f", header="Bin_Centers Counts")
    
    # Defining the bounds for the fit

    left_bound = int( (left - bin_centers[0])/ bin_size )
    right_bound = int( (right - bin_centers[0]) / bin_size )

    # Creating the expected parameters for the fit

    mean_exp = (left + right) / 2
    sigma_exp = (right - left) / 6  # Approximate sigma
    amp_exp = np.max(bins[left_bound:right_bound])

    bin_centers_fit = bin_centers[left_bound:right_bound]
    bins_fit = bins[left_bound:right_bound]

    # Fitting the Gaussian to the histogram data
    
    p0 = [amp_exp, mean_exp, sigma_exp]
    par, cov = optimize.curve_fit(gaussian, bin_centers_fit, bins_fit, p0=p0, maxfev=50000)
    a_fit, x0_fit, sigma_fit = par
    a_err, x0_err, sigma_err = np.round(np.sqrt(np.diag(cov)), decimals=2)

    print(f'x_0 fit = {x0_fit} ± {x0_err}')
    print(f'sigma fit = {sigma_fit} ± {sigma_err}')

    print(f"Fitted parameters: a = {a_fit}, x0 = {x0_fit}, sigma = {sigma_fit}")

    x = np.linspace(bin_centers_fit[0], bin_centers_fit[-1], 1000)
    y_fit = gaussian(x, a_fit, x0_fit, sigma_fit)

    # Calculating the Compton Edge

    compton_edge = x0_fit + ( 2 / 3 ) * np.abs(sigma_fit)
    print(f"Compton Edge: {compton_edge}")
    x_compton = compton_edge * np.ones(1000)
    y_compton = np.linspace(0, np.max(bins), 1000)

    # Calculating the Full Width at Half Maximum (FWHM)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma_fit

    return x, y_fit, x_compton, y_compton, compton_edge, x0_err


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Compton Edge Plotting Utility
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def plotComptonEdge(bin_centers, bins, x, y_fit, x_compton, y_compton, compton_edge, x0_err, source):

    # Plotting the histogram

    plt.plot(bin_centers, bins, color='blue', label='Total Integral Histogram')
    plt.plot(x, y_fit, 'r--', label='Gaussian Fit')
    plt.plot(x_compton, y_compton, 'g--', label=f'Compton Edge: {compton_edge:.2f} $\pm$ {x0_err} ADC Counts')

    plt.xlabel('Total Integral (ADC output)')
    plt.ylabel('Counts')
    plt.title(f'{source} Energy Spectrum with Compton Edge Fit')
    plt.legend()
    plt.grid()
    plt.show()
    

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# PSD Plotting Utitlity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def plotPSD(a_array, b_array, c_array):
    
#NOTE: This code is written assuming the C window is the toal integral, while the A and B integrals of the tail with different
#starting points. If your setup is different, change the letter numbers in the PSD definition accordingly. 

    #Calculating PSD:
    
    PSD_a = (a_array) / c_array
    PSD_b = (b_array) / c_array

    #Filtering events:

    threshold_upper = 1
    threshold_lower = -1
    threshold_left = 200
    threshold_right = 5000

    #Masking unwanted values:

    mask_a = (threshold_lower <= PSD_a) & (PSD_a <= threshold_upper) & (c_array <= threshold_right) & (c_array >= threshold_left)
    mask_b = (threshold_lower <= PSD_b) & (PSD_b <= threshold_upper) & (c_array <= threshold_right) & (c_array >= threshold_left)

    #Applying the mask:

    PSD_a_filtered = PSD_a[mask_a]
    PSD_b_filtered = PSD_b[mask_b]
    tot_int_filtered_a = c_array[mask_a]
    tot_int_filtered_b = c_array[mask_b]

    print('Filtered PSD_a mean: ', np.mean(PSD_a_filtered))
    print('Filtered PSD_b mean: ', np.mean(PSD_b_filtered))

    #Defining the subplots:

    fig, (axs_a, axs_b) = plt.subplots(1, 2)

    #Plotting the Heatmap:

    fig0 = axs_a.hist2d(tot_int_filtered_a, PSD_a_filtered, bins=500, cmap='YlOrBr')
    fig1 = axs_b.hist2d(tot_int_filtered_b, PSD_b_filtered, bins=500, cmap='YlOrBr')
    fig.colorbar(fig1[3], label='Counts')
    fig.suptitle("PSD vs. total integral")

    axs_a.set_xlabel("Total integral (ADC Counts)")
    axs_b.set_xlabel("Total integral (ADC Counts)")

    axs_a.set_ylabel("PSD")
    #axs_b.set_ylabel("PSD")

    axs_a.set_title(r"$\frac{a}{c}$")
    axs_b.set_title(r"$\frac{b}{c}$")

    fig.tight_layout

    plt.show()

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Making numpy arrays for A, B, and C from the binary file output
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def ABC(text_file):

    # Initialize lists to collect values
    a_vals, b_vals, c_vals, t_vals = [], [], [], []
    
    # Read file line by line
    with open(text_file, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Match the patterns using regex
            if line.startswith('......... A ->'):
                match = re.search(r'A -> (-?\d+)', line)
                if match:
                    a_vals.append(int(match.group(1)))
                    
            elif line.startswith('......... B ->'):
                match = re.search(r'B -> (-?\d+)', line)
                if match:
                    b_vals.append(int(match.group(1)))
                    
            elif line.startswith('......... C ->'):
                match = re.search(r'C -> (-?\d+)', line)
                if match:
                    c_vals.append(int(match.group(1)))
                    
            elif line.startswith('......... T ->'):
                match = re.search(r'T -> (-?\d+)', line)
                if match:
                    t_vals.append(int(match.group(1)))
    
    # Convert lists to NumPy arrays
    a_array = np.array(a_vals)
    b_array = np.array(b_vals)
    c_array = np.array(c_vals)
    t_array = np.array(t_vals)
    
    # Optional: print to verify
    print("A:", a_array)
    print("B:", b_array)
    print("C:", c_array)
    print("T:", t_array)

    return a_array, b_array, c_array

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Histogram Peak Finder
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def histogram_peak_finder(data): #UNFINISHED

    # Calculating a second derivative
    second_derivative = np.gradient(np.gradient(data))



# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Integral crossing Finder
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def zeroCrossing(delay_input, mean_a, mean_b, mean_c):
    
    #Making the subplots

    fig, (axs_a, axs_b, axs_c) = plt.subplots(3, 1)

    #Plotting the data

    axs_a.plot(delay_input, mean_a, 'ro', label='A')
    axs_b.plot(delay_input, mean_b, 'go', label='B')
    axs_c.plot(delay_input, mean_c, 'bo', label='C')

    fig.suptitle("ADC values vs. delay")

    axs_a.set_xlabel("Input Delay (ns)")
    axs_b.set_xlabel("Input Delay (ns)")
    axs_c.set_xlabel("Input Delay (ns)")
    axs_a.set_ylabel("ADC value")
    axs_b.set_ylabel("ADC value")
    axs_c.set_ylabel("ADC value")

    axs_a.legend()
    axs_b.legend()
    axs_c.legend()

    fig.tight_layout

    plt.show()
    return()


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  MAIN PROGRAM
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Read all .bin files from a folder.")
    parser.add_argument("folderPath", help="Path to the folder containing .bin files")
    args = parser.parse_args()

    # Make sure folder exists
    if not os.path.isdir(args.folderPath):
        print(f"Error: Folder '{args.folderPath}' does not exist.")
        exit(1)

    # Loop through each .bin file in the folder
    bin_files = glob.glob(os.path.join(args.folderPath, '*.bin'))

    if not bin_files:
        print("No .bin files found in the folder.")
        exit(1)

    # Finding means of the A, B, and C arrays

    mean_a_list, mean_b_list, mean_c_list = [], [], []

    for bin_file in sorted(bin_files):
        print(f"\nReading {os.path.basename(bin_file)}\n")

        with open(bin_file, "rb") as fid_bin:
            get_event_packets(fid_bin)

        # Generate output filename for each bin file
        output_filename = f"{os.path.splitext(os.path.basename(bin_file))[0]}_readable_output.txt"

        # Write human-readable output for each bin file
        with open(output_filename, 'w') as outfile:
            with contextlib.redirect_stdout(outfile):
                with open(bin_file, "rb") as fid_bin:
                    get_event_packets(fid_bin)

        # Extract arrays and calculate means
        a_array, b_array, c_array = ABC(output_filename)
        mean_a = np.mean(a_array)
        mean_b = np.mean(b_array)
        mean_c = np.mean(c_array)

        mean_a_list.append(mean_a)
        mean_b_list.append(mean_b)
        mean_c_list.append(mean_c)

        print(f"Means for {os.path.basename(bin_file)}: A={mean_a}, B={mean_b}, C={mean_c}")

    output_filename = "event_data_readable_output.txt"

    with open(output_filename, 'w') as outfile:
        with contextlib.redirect_stdout(outfile):
            for bin_file in sorted(bin_files):
                print(f"\nReading {os.path.basename(bin_file)}\n")
                with open(bin_file, "rb") as fid_bin:
                    get_event_packets(fid_bin)

    print(f"\nAll .bin files processed. Human-readable output saved to {output_filename}")

    #Looping through each .json file in the folder

    json_files = glob.glob(os.path.join(args.folderPath, '*.json'))

    if not json_files:
        print("No .json files found in the folder.")
    else:
        delay_input_list = []

        for json_file in sorted(json_files):
            print(f"\nReading {os.path.basename(json_file)}\n")
            delay_input_list.append(jsonReader(json_file))

        delay_input = np.array(delay_input_list)

    # Function Call for PSD plot

    a_array, b_array, c_array = ABC(output_filename)
    #plotPSD(a_array, b_array, c_array)

    # Histogram Binning and Compton Edge Plotting

    # Defining fit range (EDITH THIS FOR YOUR SOURCE)

    left = 8000
    right = 10000
    source = "Co60"  # Change this to your source name

    bin_centers, bins, bin_size = histogramBinning(c_array)
    x, y_fit, x_compton, y_compton, compton_edge, x0_err = gaussFit(bin_centers, bins, bin_size, left, right, source)
    plotComptonEdge(bin_centers, bins, x, y_fit, x_compton, y_compton, compton_edge, x0_err, source)

    # Function call for delay plot

    #mean_a = np.array(mean_a_list)
    #mean_b = np.array(mean_b_list)
    #mean_c = np.array(mean_c_list)
    #zeroCrossing(delay_input, mean_a, mean_b, mean_c)


    print("\nAll files processed. Exiting...\n")
    exit()

