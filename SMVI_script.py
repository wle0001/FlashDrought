# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:40:47 2024

@author: sdeal/ellenbw
"""

import rasterio as rio
#import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
#import sys
#import gc
from datetime import timedelta
import datetime
sys.exit()
print(datetime.datetime.now())
RsList = glob.glob(r'LIS/SPoRT-LIS_data/SM_clipped_weekly_file_list/*.tif')
RsList.sort()
#RsList = RsList[6000:]
#sys.exit()
# Read in the first file
tif1meta = rio.open(RsList[0]).meta
tif1 = rio.open(RsList[0]).read()

# Pulling out dates for later indexing

dts = [RsList[0].split('/')[-1].split('_')[-2][:-2]]

# Stacking the Rasters
for ras in RsList[1:]:
    tif1 = np.append(tif1, rio.open(ras).read(), axis=0)
    dts.append(ras.split('/')[-1].split('_')[-2][:-2])
print(datetime.datetime.now())    
# Defining NaN values

sys.exit()
#tif1[tif1 > 1.79769313e+307] = np.nan
tif1[tif1 == 9999.] = np.nan

# Establishing blank array for image creation
flashTif = np.empty((len(dts),tif1[0].shape[0], tif1[0].shape[1]))
flashTif[:] = np.nan

# Create a Dataframe, with index of dates to create a DateTime

df = pd.DataFrame(index = pd.to_datetime(dts))
df.index = pd.to_datetime(df.index, format = '%Y%m%d')

# sys.exit()

#%%
'''

##
###get weekly means from daily data
##

print(datetime.datetime.now())
RsList = glob.glob(r'LIS/SPoRT-LIS_data/SM_daily_file_list/*.tif')
RsList.sort()

#sys.exit()
# Read in the first file
tif1meta = rio.open(RsList[0]).meta
tif1 = rio.open(RsList[0]).read()

# Pulling out dates for later indexing

dts = [RsList[0].split('/')[-1].split('_')[-2][:-2]]

# Stacking the Rasters
for ras in RsList[1:]:
    tif1 = np.append(tif1, rio.open(ras).read(), axis=0)
    dts.append(ras.split('/')[-1].split('_')[-2][:-2])
print(datetime.datetime.now())    
# Defining NaN values

sys.exit()
tif1[tif1 == 9999.] = np.nan


xray = xr.DataArray(tif1)
dray = pd.to_datetime(dts) 
dst = xray.expand_dims(time=dray)
dst = dst.assign_coords({'dim_0': dray})
dst = dst.drop('time')
dst = dst[0,:,:,:]
#gc.collect()
#offset to match up with the previous weekly files 
dst2 = dst.resample(dim_0='7d', closed = 'right',label = 'right', offset = '3d').mean()

dst3 = dst2.to_numpy()

for i in range(dst3.shape[0]):
    
    date = pd.to_datetime(dst2.coords['dim_0'][i].values).strftime('%Y%m%d12')
    
    with rio.open('new_week/percentile_{}_clipped.tif'.format(date), 'w',**tif1meta) as dest:
         dest.write(np.expand_dims(dst3[i,:,:],0))
    

'''


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
        
        df3.columns = ['7d']    # One day column
        #df3.index = pd.to_datetime(dst.dim_0)   # Making it a datetime. NOTE: DST.DIM_0 MAY NEED TO BE CHANGED TO DST2
        df3.index = pd.to_datetime(df.index)
        #df3 = df3.resample('1d').mean()
        #df3['7d'] = df3['1d'].rolling('7d').mean()
        df3['3w'] =df3['7d'].rolling('21d').mean()  # Finding the 3 week average
        df3['dif'] = df3['7d'] - df3['3w']      # Finding which one was less

        df3['state'] = 0    # I think this is "are we in a drought or not
        df3['cond1'] = 0       # Condition 1
        df3['cond2'] = 0   
        df3['cond3'] = 0# Condition 2
        pixel_state = {}    

        # This is making sure the flash droughts aren't repeating the same event
        pixel_duration2 = 0     # This one too
        occurrence = 0          # A boolean for whether a flash drought is active or not
        #occurrence_met = False
        #i = 0
        end_fd = 0
        fd_dur = 0
        #sys.exit()
    
    
        for index, rows in df3.iterrows():
                
            #pixel_state[index] = 0
            #print(index)
            if rows['dif'] < 0:
                fd_dur +=1
                df3.loc[index,'cond1'] = fd_dur
                if rows['7d'] < 20 and fd_dur >0:  #threshold met
                    end_fd = 0
                    pixel_duration2 +=1
                    df3.loc[index,'cond2'] = pixel_duration2 
                    if pixel_duration2 ==3:
                        df3.loc[index,'state'] = 1 #2 week duration met
                        for d in range(fd_dur):
                            df3.loc[index - timedelta(days=(d*7)),'state'] = 1 #If weekly, change to d*7 in days=
                    if pixel_duration2 >3:
                        df3.loc[index,'state'] = 1 #contunue FD until end_df = 2
                    
            else:
                
                end_fd +=1 #no longer meeting thres
                if end_fd >1:
                    fd_dur = 0
                else:
                    fd_dur +=1
                df3.loc[index,'cond3'] = end_fd 
                if pixel_duration2 >2: #i.e. we are coming off a flash drought
                    if end_fd < 2  : #end fd thresh not met
                        pixel_duration2 +=1 #keep counting fd
                        df3.loc[index,'state'] = 1 #still in fd
                    else: #thresh not met for two weeks in a row
                        pixel_duration2 = 0 #end FD
                        for d in range(end_fd): 
                            df3.loc[index - timedelta(days=(d*7)),'state'] = 0 # this time and previous week no longer classifed as FD
                else:
                    df3.loc[index,'state'] = 0 
                    pixel_duration2 = 0
                

            #i+=1
        df3 = df3[:len(ts)] # taking off any new dates that might have been added on- this is due to Jan/Feb missing from the data             
    #df.loc[index,'state'] = pixel_state[index]
        #df3.loc[index,'d1'] = end_fd
        #df3.loc[index,'d2'] = pixel_duration2
        

        flashTif[:,row,col] = df3.state
    
    print(dfcol,df3.state.max())
    print(datetime.datetime.now())  
#sys.exit()   
tif1meta['count'] = len(dts)     
with rio.open('smvi_FLASH.tif', 'w',**tif1meta) as dst:
     #dst.write(vic)
     dst.write(flashTif)
     
     
#%%     
    
#full counts    

FlashTif = rio.open('smvi_FLASH.tif').read()



#selectif = FlashTif
#dts = pd.date_range('2000-1-21',periods = len(RsList))
dts = pd.read_csv('smvi_falsh_index.csv', index_col = 0).index
dts = pd.to_datetime(dts)
dfts = pd.DataFrame(range(len(RsList)), index = pd.to_datetime(dts), columns = ['ids'])


#%%
#for season in ['autumn', 'winter', 'growing','spring', 'summer']:
#for mo in range(12):
for yr in range(2000,2023):
    season = 'year'
    name, select, selectif = seasonSel(season,dfts, FlashTif, yr)
    #select = dfts
    
    flashTifcount = np.empty((selectif[0].shape[0], selectif[0].shape[1]))
    flashTifcount[:] = np.nan
    
    for row in range(selectif[0].shape[0]):
        for col in range(selectif[0].shape[1]):
            dfcol = str(row)+'_'+str(col)
    
            ts = pd.Series(selectif[:,row, col], index = select.index, name = dfcol)
            
            conds = []
            for ix, fd in enumerate(ts):
            
                if fd == 1:
                    if fdcount == 0:
                        val = 1
                        conds.append(val)
                    else:
                        val = 0
                        conds.append(val)
                    fdcount = 1
                else:
                    fdcount = 0
                    val = 0
                    conds.append(val)
                    #if 1 in conds[ix-4:ix]:
                        #val = 0 # flash drought already counted
                    #else:
                        #val = 1   #3 = flash drought
                    #val = 3
                    #conds.append(val)
                #else:
                    #val = np.nan
                    #conds.append(val)
                    
            #DF3[dfcol] = conds
            conds = pd.Series(conds, index = ts.index)
            
            count = len(conds[conds==1])
            flashTifcount[row,col] = count
            
         
    tif1meta['count'] = 1
    with rio.open('SMVI_total_count_{}.tif'.format(name), 'w', **tif1meta) as dst:
        dst.write(np.expand_dims(flashTifcount,0)) 
#%%    
def seasonSel(season, dfts, tif1, my):   

    tif_name = 'dm_falshCount_{}.tif'.format(season)
    if season == 'month':
        tif_name = 'dm_falshCount_{}_{}.tif'.format(season,str(my+1))
    if season == 'year':
        tif_name = 'dm_falshCount_{}_{}.tif'.format(season,str(my))
    
    if season == 'year':
        select = dfts[(dfts.index.year == yr)]['ids']
        selectif = tif1[select,:,:]
    elif season == 'month':
        select = dfts[(dfts.index.month == mo+1)]['ids']
        selectif = tif1[select,:,:] 
    elif season == 'all':
        select = dfts
        selectif = tif1
    elif season == 'spring':
        select = dfts[(dfts.index.month == 3)|(dfts.index.month == 4)|(dfts.index.month == 5)]['ids']
        selectif = tif1[select,:,:] 
    elif season == 'summer':
        select = dfts[(dfts.index.month == 6)|(dfts.index.month == 7)|(dfts.index.month == 8)]['ids']
        selectif = tif1[select,:,:] 
    elif season == 'autumn':
        select = dfts[(dfts.index.month == 9)|(dfts.index.month == 10)|(dfts.index.month == 11)]['ids']
        selectif = tif1[select,:,:]
    elif season == 'winter':
        select = dfts[(dfts.index.month == 12)|(dfts.index.month == 1)|(dfts.index.month == 2)]['ids']
        selectif = tif1[select,:,:] 
    elif season == 'growing':
        select = dfts[(dfts.index.month == 5)|(dfts.index.month == 6)|(dfts.index.month == 7)|(dfts.index.month == 8)|(dfts.index.month == 9)]['ids']
        selectif = tif1[select,:,:] 
    name = season    
    if season == 'month':
        name = season+str(my+1)
    if season == 'year':
        name = season+str(my)
    return name, select, selectif
#%%
''' for plotting df3 within the loop # need to have tif1 compiled
fig, ax = plt.subplots()
ax2 = ax.twinx()

df3[['7d','3w']].plot(ax = ax, style='.-')
df3['20pct'] = 20
df3['20pct'].plot(ax = ax)

ax.fill_between(df3.index, 0,1, where=df3['state'], transform=ax.get_xaxis_transform(), color = 'brown', alpha  = 0.5)
'''
#%%


import geopandas as gpd
from rasterstats import zonal_stats
states = gpd.read_file('/Users/ellenbw/Documents/UAH/SCO/NOAA_FD/Shae/SE_states.shp')
shp = '/Users/ellenbw/Documents/UAH/SCO/LGI/HUC12_SE/SEHuc12.shp'
gdf = gpd.read_file(shp)
#%%
tif = 'SMVI_total_count_month6.tif'

gdf['mean'] = (pd.DataFrame(
    zonal_stats(
        vectors=gdf['geometry'], 
        raster=tif, 
        stats='mean'
    )
)['mean']/22)*100

fig, ax = plt.subplots() 

gdf.plot(column='mean', cmap='hot_r',vmax = 100,legend=True, ax = ax) #,vmax = 5
states.plot(facecolor = 'none', ax= ax)


#%%
#poststamp plots
titles = ['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug', 'Sep','Oct','Nov','Dec']
#titles = range(2000,2023)
states = gpd.read_file('/Users/ellenbw/Documents/UAH/SCO/NOAA_FD/Shae/SE_states.shp')
shp = '/Users/ellenbw/Documents/UAH/SCO/LGI/HUC12_SE/SEHuc12.shp'
gdf = gpd.read_file(shp)
#%%
#tifs = glob.glob('dm_falshCount_month*.tif').sort()
i = 0
fig, axs = plt.subplots(3,4)

for axv in axs:
    for axh in axv:
        
        tif = 'SMVI_total_count_month{}.tif'.format(i+1)
    
        gdf['mean'] = pd.DataFrame(
            zonal_stats(
                vectors=gdf['geometry'], 
                raster=tif, 
                stats='mean'
            )
        )['mean']
        
        
       
        gdf.plot(column='mean', cmap='hot_r', legend=False, ax = axh) #vmax = 5,
        states.plot(facecolor = 'none', ax= axh)
       
        axh.set_title(str(titles[i]))
        i+=1
        
        
sm = plt.cm.ScalarMappable(cmap='hot_r', norm=plt.Normalize(vmin=0, vmax=5))        
cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])    

fig.colorbar(sm, cax = cax)


        