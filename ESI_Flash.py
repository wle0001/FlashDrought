# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:05:04 2024

@author: sdeal
"""

import rasterio as rio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import sys
import gc
from datetime import timedelta
import datetime

RsList = glob.glob(r'C:\Users\shael\Documents\AlexiClipped\*.tif')
RsList.sort()

# Read in the first file
tifmeta = rio.open(RsList[0]).meta
tif1 = rio.open(RsList[0]).read()

# Pulling out dates for later indexing

dts = [RsList[0].split('\\')[-1][:-4]]
# dts = datetime.datetime.strptime(dts, '%y%j')

# Stacking the Rasters
for ras in RsList[1:]:
    tif1 = np.append(tif1, rio.open(ras).read(), axis=0)
    dts.append(ras.split('/')[-1].split('\\')[-1][:-4])

tif1[tif1 < -3.6] = np.nan


df = pd.DataFrame(index = dts)
df.index = pd.to_datetime(df.index, format = '%Y%j')

flashTif = np.empty((tif1[0].shape[0], tif1[0].shape[1]))
flashTif[:] = np.nan

#%%


counts = []    # Number of flash droughts
occurrence = 0

fd = []
#prev = 0


#ts = dst3[:,row,col]    # Dataframe of each file
ts = tif1[:,50,50]    # Dataframe of each file

df3 = pd.DataFrame(ts)  # Defining the dataframe

df3.columns = ['7d']
df3.index = pd.to_datetime(df.index)
df3['4w'] =df3['7d'].rolling('28d').mean()

df3['change'] = df3['4w'].rolling(window=3).apply(lambda df3: df3.iloc[2] - df3.iloc[0])

#%%

# This chunk of code is all speculation.
negative_sequences = []
current_sequence = []

for val in df3['change']:
    if val < 0:
        current_sequence.append(val)
    elif current_sequence:
        negative_sequences.append(current_sequence)
        current_sequence = []
if current_sequence:
    negative_sequences.append(current_sequence)

# Step 2 & 3: Mark 'fd' for sequences with values <= -0.8
fd = [0] * len(df3['change'])
for sequence in negative_sequences:
    if any(val <= -0.8 for val in sequence):
        start_index = None
        for i, val in enumerate(df3['change']):
            if val == sequence[0]:
                start_index = i
                break
        if start_index is not None:
            end_index = start_index + len(sequence)
            fd[start_index:end_index] = [1] * len(sequence)

df3['pos'] = fd


#%%

# below is code that I know works
# for index, rows in df3.iterrows():

#     if rows['change'] < 0:
#         occurrence = 1
#         counts.append(occurrence)

#     else:
#         occurrence = 0
#         counts.append(occurrence)
        
# #count_ones = counts.count(1)

new = []
counts = df3['pos'].tolist()

for i in range(len(counts)):
    if counts[i] == 1:
        if i == 0 or i == len(counts)-1:
            new.append(0)
        elif counts[i-1] == 0 and counts[i+1] == 0:
            new.append(0)
        elif counts[i-1] == 1 and counts[i] == 1:
            new.append(0)
        else:
            new.append(1)
    else:
        new.append(0)


sumstuff = float(sum(new))

# df3['pos'] = counts
df3['fd'] = new
#%%        
fig, ax = plt.subplots()
#df3[['7d']].plot(ax = ax, style='.-', linewidth=3, label='7d')
df3[['4w']].plot(ax = ax, style='.-', color = 'orange', linewidth=1, label='3w')
df3[['change']].plot(ax = ax, style='-', color='green', linewidth=1, label='dif')
df3[['pos']].plot(ax = ax, style='.-', color = 'blue', linewidth=3, label='active drought')
df3[['fd']].plot(ax = ax, style='.', color = 'red', linewidth=6, label='FD')

ax.set_title('Flash Droughts Identified for Pixel: 50,50', fontsize=16)
ax.grid(True)
ax.set_xlabel('Year')


# Go back and set a condition that it only catches fd2 as a flash drought if at some point during the 1 sequence it falls below -0.84
