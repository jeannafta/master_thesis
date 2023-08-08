#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from scipy.interpolate import interp1d

import serial as ser
import csv
import os

from datetime import date
from datetime import datetime

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arduino_port', type=str, default="/dev/ttyACM0", help='Arduino port ACM0 or ACM1.')
    # to give permissions with linux terminal: sudo chmod a+rw /dev/ttyACM0
    args = parser.parse_args()

    #########################################################################################################
    ########################## Resistance-temperature relation for each thermistor ##########################
    #########################################################################################################

    # Fluid temperature transducer

    table_fluid = np.genfromtxt('transducers/Table_R-T_fluid_T_transducer.csv', delimiter=',', skip_header=1)
    x, y = table_fluid[:, 1], table_fluid[:, 0]
    fit_fluid = interp1d(x, y, 'cubic')    # compute LMS cubic regression

    #### Surface temperature transducer

    table_surface = np.genfromtxt('transducers/Table_R-T_surface_T_transducer.csv', delimiter=',', skip_header=1)
    x, y = table_surface[:, 1], table_surface[:, 0]
    fit_surface = interp1d(x, y, 'cubic')    # compute LMS cubic regression

    #########################################################################################################
    ############################### Connection to Arduino and data collection ###############################
    #########################################################################################################

    # Connect to Arduino port
    serial = ser.Serial(args.arduino_port, 9600)
    print("Connected to Arduino port")

    # Create a folder for today's data if the folder does not already exist
    folder = "analog_data/"+str(date.today())
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Directory " + folder + "created")

    # Open the file and write data
    file_name = str(date.today())+"_"+datetime.now().strftime("%H:%M:%S")+".csv"

    try:
        with open(folder+"/"+file_name, 'w', newline='') as file:

            # write data in a csv file
            writer = csv.writer(file, delimiter=',')
            # write header
            writer.writerow(["Time", "Resistance outlet water", "Resistance tubing outlet", "Resistance tubing inlet", "Resistance silicone oil",
                             "Temperature outlet water", "Temperature tubing outlet", "Temperature tubing inlet", "Temperature silicone oil"])
            
            # create lists to plot temperatures
            list1, list2, list3, list4, list5 = [], [], [], [], []

            # write datapoint and update plot
            i = 1
            plt.ion()
            fig = plt.figure()
            while True:
                # write datapoint in the csv
                row = serial.readline().decode("utf-8")
                R1 = float(row.split(",")[0])
                R2 = float(row.split(",")[1])
                R3 = float(row.split(",")[2])
                R4 = float(row.split(",")[3])
                               
                writer.writerow([datetime.now().strftime("%H:%M:%S"), str(R1), str(R2), str(R3), str(R4), 
                                 str(fit_surface(R1)), str(fit_surface(R2)), str(fit_surface(R3)), str(fit_fluid(R4))])
                
                # plot collected data
                list1.append(i)
                list2.append(fit_surface(R1))
                list3.append(fit_surface(R2))
                list4.append(fit_surface(R3))
                list5.append(fit_fluid(R4))
                plt.plot(list1, list2, 'k', label="outlet water")
                plt.plot(list1, list3, 'b', label="tubing outlet")
                plt.plot(list1, list4, 'r', label="tubing inlet")
                plt.plot(list1, list5, 'm', label="silicone oil")
                if i==1: 
                    plt.legend(loc="upper left")
                    plt.xlabel("Time [s]")
                    plt.ylabel("Temperature [°C]")
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                print("#########")
                print(fit_surface(R1))
                print(fit_surface(R2))
                print(fit_surface(R3))
                print(fit_fluid(R4))
                
                i+=1

    except KeyboardInterrupt:    
        print("Data collection is completed")
        
        
if __name__ == '__main__':
    main()
