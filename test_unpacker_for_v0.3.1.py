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
# PSD Plotting Utitlity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def plotPSD(text_file):

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
    
    #Calculating PSD and plotting:
    
    PSD = (a_array - b_array) / a_array
    
    plt.plot(a_array, PSD, 'bo')
    plt.title("PSD vs. total integral")
    plt.xlabel("Total integral (ADC Counts)")
    plt.ylabel("PSD")
    plt.show()



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

    for bin_file in sorted(bin_files):
        print(f"\nReading {os.path.basename(bin_file)}\n")

        with open(bin_file, "rb") as fid_bin:
            get_event_packets(fid_bin)

    import contextlib

    output_filename = "event_data_readable_output.txt"

    with open(output_filename, 'w') as outfile:
        with contextlib.redirect_stdout(outfile):
            for bin_file in sorted(bin_files):
                print(f"\nReading {os.path.basename(bin_file)}\n")
                with open(bin_file, "rb") as fid_bin:
                    get_event_packets(fid_bin)

    print(f"\nAll files processed. Human-readable output saved to {output_filename}")

    plotPSD(output_filename)

    print("\nAll files processed. Exiting...\n")
    exit()

