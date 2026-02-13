#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  This script is used to validate the python implementation of 
  Recommendation ITU-R P.528 as defined in the package Py528
  
  Author: Ivica Stevanovic (IS), Federal Office of Communications, Switzerland
 
  Revision History:
  Date            Revision
  13FEB26       Initial version (IS)
"""

import os
import re
import glob
from pathlib import Path


import pandas as pd
import numpy as np

from Py528 import P528

tol = 0.1 # Tolerance in dB, the precision in reference files is 0.1

# path to the folder containing test profiles
test_profiles = "./Data_Tables/"

# Collect all the filenames .csv in the folder test_profiles
try:
    filenames = [f for f in os.listdir(test_profiles) if f.endswith(".csv")]
except:
    print("The system cannot find the given folder " + test_profiles)

cnt_fail = 0
cnt_pass = 0

total_files = len(filenames)

for idx, filename in enumerate(filenames, start=1):
    print(f"\nProcessing file {idx}/{total_files}: {filename}")
    # Read the file
    with open(test_profiles + filename, 'r') as fid:
        lines = fid.readlines()

    # Extract frequency and percentage from first line
    first_line = lines[0].strip()
    # Parse "1200MHz / Lb(0.01) dB"
    f= int(first_line.split('MHz')[0])  # Extract 1200
    p = float(first_line.split('(')[1].split(')')[0])  # Extract 0.01

    # Extract h2 from line 2 (index 1)
    h2_line = lines[1].strip().split(',')
    h2 = np.array([float(x) for x in h2_line[2:] if x])  # Skip first two elements, filter empty strings

    # Extract h1 from line 3 (index 2)
    h1_line = lines[2].strip().split(',')
    h1 = np.array([float(x) for x in h1_line[2:] if x])  # Skip first two elements, filter empty strings

    # Read the actual data starting from line 5 (index 4)
    # Skip first 4 lines
    df = pd.read_csv(test_profiles + filename, skiprows=4)

    # Extract distances (first column)
    distance_col = df.columns[0]
    d = df[distance_col].values

    # Extract path loss matrix (all columns except first two)
    # Skip the first column (distances) and second column (FSL)
    Lb_ref = df.iloc[:, 2:].values

    # Convert d and path_loss to proper types
    d = d.astype(float)
    Lb_ref = Lb_ref.astype(float)

    Lb_computed = []
    
    # Process every 1000th distance value
    print(f'{"Python":>20s} {"REF TABLE":>20s} {"DELTA":>20s}')
    for i in range(0, len(d), 1000):
        for j in range(len(h1)):
            # Call tl_p528 function
            result = P528.bt_loss(d[i], h1[j], h2[j], f, 0, p * 100)
            delta = round(10.0 * (result.A_db - Lb_ref[i][j])) / 10.0
            
            if abs(delta) > 0.1:
                cnt_fail += 1
            else:
                cnt_pass += 1
            
            print(f'{result.A_db:>20.1f} {Lb_ref[i][j]:>20.1f} {delta:>20.1f}')

print(f'Successfully passed {cnt_pass} out of {cnt_pass + cnt_fail} tests')