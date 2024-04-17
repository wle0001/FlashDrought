# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:40:47 2024

@author: sdeal
"""

import rasterio as rio
#import xarray as xr
import numpy as np
#import matplotlib.pyplot as plt
import glob
import pandas as pd
#import sys
#import gc
from datetime import timedelta
import datetime
print(datetime.datetime.now())
RsList = glob.glob(r'geoTIF/LGI_*.tif')
RsList.sort()
RsList = RsList[6000:]
#sys.exit()
# Read in the first file
tif1meta = rio.open(RsList[0]).meta
tif1 = rio.open(RsList[0]).read()

# Pulling out dates for later indexing

dts = [RsList[0].split('/')[-1].split('_')[-2]]

# Stacking the Rasters
for ras in RsList[1:]:
    tif1 = np.append(tif1, rio.open(ras).read(), axis=0)
    dts.append(ras.split('/')[-1].split('_')[-2])
print(datetime.datetime.now())    
# Defining NaN values
tif1[tif1 > 1.79769313e+307] = np.nan

# Establishing blank array for image creation
flashTif = np.empty((len(dts),tif1[0].shape[0], tif1[0].shape[1]))
flashTif[:] = np.nan

# Create a Dataframe, with index of dates to create a DateTime

df = pd.DataFrame(index = dts)
df.index = pd.to_datetime(df.index, format = '%Y%m%d')

# sys.exit()

#%%

# Create an xarray thing to resample the tifs into weekly data.
# This is ONE possibility, hence it being in its own sad little cell

# xray = xr.DataArray(tif1)
# dray = pd.date_range(RsList[0].split('/')[-1].split('_')[-2], RsList[-1].split('/')[-1].split('_')[-2])

# dst = xray.expand_dims(time=dray)
# dst = dst.assign_coords({'dim_0': dray})
# dst = dst.drop('time')
# dst = dst[0,:,:,:]
# gc.collect()
# # dst2 = dst.resample(dim_0='1W', closed = 'right', label = 'right').mean()
# # dst3 = dst2.to_numpy()
# dst3 = dst.to_numpy()

# sys.exit()

#%%

# THIS cell is a continuation of the last one, in which the dataset has been resampled to Weeks
df4 = pd.DataFrame()
for row in range(tif1[0].shape[0]):
    for col in range(tif1[0].shape[1]):
        dfcol = str(row)+'_'+str(col)
    
        counts = [0]    # Number of flash droughts
        
        #ts = dst3[:,row,col]    # Dataframe of each file
        ts = tif1[:,row,col]    # Dataframe of each file
    
        df3 = pd.DataFrame(ts)  # Definint the dataframe
        
        df3.columns = ['1d']    # One day column
        #df3.index = pd.to_datetime(dst.dim_0)   # Making it a datetime. NOTE: DST.DIM_0 MAY NEED TO BE CHANGED TO DST2
        df3.index = pd.to_datetime(df.index)
        df3 = df3.resample('1d').mean()
        df3['7d'] = df3['1d'].rolling('7d').mean()
        df3['3w'] =df3['1d'].rolling('21d').mean()  # Finding the 3 week average
        df3['dif'] = df3['7d'] - df3['3w']      # Finding which one was less

        df3['state'] = 0    # I think this is "are we in a drought or not
        df3['d1'] = 0       # Condition 1
        df3['d2'] = 0       # Condition 2
        pixel_state = {}    

        # This is making sure the flash droughts aren't repeating the same event
        pixel_duration2 = 0     # This one too
        occurrence = 0          # A boolean for whether a flash drought is active or not
        #occurrence_met = False
        #i = 0
        end_fd = 0
        #sys.exit()
    
    
        for index, rows in df3.iterrows():
                
            #pixel_state[index] = 0
            #print(index)
            if rows['dif'] < 0 and rows['7d'] < 0:  
                end_fd = 0
                pixel_duration2 +=1
                if pixel_duration2 ==7:
                    df3.loc[index,'state'] = 1
                    for d in range(pixel_duration2):
                        df3.loc[index - timedelta(days=(d)),'state'] = 1 #If weekly, change to d*7 in days=
                if pixel_duration2 >7:
                    df3.loc[index,'state'] = 1
                    
            else:
                end_fd +=1
                if pixel_duration2 >6: #i.e. we are coming off a flash drought
                    if end_fd <7 :
                        pixel_duration2 +=1
                        df3.loc[index,'state'] = 1
                    else:
                        pixel_duration2 = 0
                        for d in range(end_fd):
                            df3.loc[index - timedelta(days=(d)),'state'] = 0
                else:
                    df3.loc[index,'state'] = 0
                

            #i+=1
                    
    #df.loc[index,'state'] = pixel_state[index]
        df3.loc[index,'d1'] = end_fd
        df3.loc[index,'d2'] = pixel_duration2
        
        '''
        for i in range(1, len(df3)-1):
            if df3.iloc[i]['state'] == 1 and df3.iloc[i-1]['state'] == 0 and df3.iloc[i+1]['state'] == 1:
                occurrence = 1
                counts.append(occurrence)
            else:
                occurrence = 0
                counts.append(occurrence)
            
            sumstuff = float(sum(counts))
            
            counts.append(0)
            
            flashTif[row,col] = sumstuff
        '''

        flashTif[:,row,col] = df3.state
    
    print(dfcol,df3.state.max())
    print(datetime.datetime.now())  
#sys.exit()   
tif1meta['count'] = len(dts)     
with rio.open('test2.tif', 'w',**tif1meta) as dst:
     #dst.write(vic)
     dst.write(flashTif)
     
     
#%%     
t1 = rio.open('subTest/sub1.tif')
t2 = rio.open('subTest/sub2.tif')
t3 = rio.open('subTest/sub3.tif')
t4 = rio.open('subTest/sub4.tif')

FlashTif = np.vstack([t1.read(), t2.read()[8:,:,:], t3.read()[8:,:,:], t4.read()[8:,:,:]])


tif1meta['count'] = len(flashTif)  
with rio.open('LGI_FLASH.tif', 'w',**tif1meta) as dst:
     #dst.write(vic)
     dst.write(flashTif)
     
     
#full count     

selectif = FlashTif
dts = pd.date_range('2000-1-21',periods = len(RsList))
dfts = pd.DataFrame(range(len(RsList)), index = pd.to_datetime(dts), columns = ['ids'])

select = dfts

flashTifcount = np.empty((selectif[0].shape[0], selectif[0].shape[1]))
flashTifcount[:] = np.nan

for row in range(selectif[0].shape[0]):
    for col in range(selectif[0].shape[1]):
        dfcol = str(row)+'_'+str(col)

        ts = pd.Series(selectif[:,row, col], index = select.index, name = dfcol)
        
        conds = []
        for ix, fd in enumerate(ts):
        
            if fd == 1:
                if 1 in conds[ix-4:ix]:
                    val = 0 # flash drought already counted
                else:
                    val = 1   #3 = flash drought
                #val = 3
                conds.append(val)
            else:
                val = np.nan
                conds.append(val)
                
        #DF3[dfcol] = conds
        conds = pd.Series(conds, index = ts.index)
        
        count = len(conds[conds==1])
        flashTifcount[row,col] = count

with rio.open('LGI_total_count.tif', 'w', **tif1meta) as dst:
    dst.write(np.expand_dims(flashTif,0)) 
        
        
    
        
    
 #test   
'''

dfcol = str(row)+'_'+str(col)

counts = [0]    # Number of flash droughts

#ts = dst3[:,row,col]    # Dataframe of each file
ts = tif1[:,row,col]    # Dataframe of each file

df3 = pd.DataFrame(ts)  # Definint the dataframe

df3.columns = ['1d']    # One day column
#df3.index = pd.to_datetime(dst.dim_0)   # Making it a datetime. NOTE: DST.DIM_0 MAY NEED TO BE CHANGED TO DST2
df3.index = pd.to_datetime(df.index)
df3 = df3.resample('1d').mean()
df3['7d'] = df3['1d'].rolling('7d').mean()
df3['3w'] =df3['1d'].rolling('21d').mean()  # Finding the 3 week average
df3['dif'] = df3['7d'] - df3['3w']      # Finding which one was less

df3['state'] = 0    # I think this is "are we in a drought or not
df3['d1'] = 0       # Condition 1
df3['d2'] = 0       # Condition 2
pixel_state = {}    

# This is making sure the flash droughts aren't repeating the same event
pixel_duration2 = 0     # This one too
occurrence = 0          # A boolean for whether a flash drought is active or not
#occurrence_met = False
#i = 0
end_fd = 0
#sys.exit()


for index, rows in df3.iterrows():
        
    #pixel_state[index] = 0
    #print(index)
    if rows['dif'] < 0 and rows['7d'] < 0:  
        end_fd = 0
        pixel_duration2 +=1
        if pixel_duration2 ==7:
            df3.loc[index,'state'] = 1
            for d in range(pixel_duration2):
                df3.loc[index - timedelta(days=(d)),'state'] = 1 #If weekly, change to d*7 in days=
        if pixel_duration2 >7:
            df3.loc[index,'state'] = 1
            
    else:
        end_fd +=1
        if pixel_duration2 >6: #i.e. we are coming off a flash drought
            if end_fd <7 :
                pixel_duration2 +=1
                df3.loc[index,'state'] = 1
            else:
                pixel_duration2 = 0
                for d in range(end_fd):
                    df3.loc[index - timedelta(days=(d)),'state'] = 0
        else:
            df3.loc[index,'state'] = 0
'''    
        