# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:39:32 2023

@author: sdeal
"""

# direc = r'\\uahdata\rhome\NewClipped\*.tif'
# season = spring, summer, autumn, winter, growing

  
import rasterio as rio
import xarray as xr
#import rioxarray as riox
import numpy as np

from scipy.interpolate import griddata
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
import pandas as pd
import sys 
from datetime import datetime
from matplotlib.dates import date2num



# hello
direc = '../DM_data/NewClipped/*.tif'

yr = ''
mo = ''
cond = 1#123

if cond == 1: #2cat/3weeks
    period = 3
    catchange = 1 # > cat change so 1 is a 2 or more
    outdir = '../DM_FD_Cond{}/'.format(cond)
    
if cond == 2: #3cat/5weeks
    period = 5
    catchange = 2 # > cat change so 2 is a 3 or more
    outdir = '../DM_FD_Cond{}/'.format(cond)
    
if cond == 3: #4cat/6weeks
    period = 6
    catchange = 3 # > cat change so 3 is a 4 or more
    outdir = '../DM_FD_Cond{}/'.format(cond)
    

# Call correct data from folder. add \*.tif to the end of it
tifList = glob.glob(direc)
# Here we are getting the dates from the titles of the files
tifList.sort()
tif1 = rio.open(tifList[0]).read()


tifmeta = rio.open(tifList[0]).meta
dts = [tifList[0].split('/')[-1].split('_')[-1][:8]]

for tifp in tifList[1:]:
    
    dts.append(tifp.split('/')[-1].split('_')[-1][:8])
    
    tif = rio.open(tifp).read()
    
    tif1 = np.append(tif1,tif, axis = 0)
    
dfts = pd.DataFrame(range(len(dts)), index = pd.to_datetime(dts), columns = ['ids'])
#sys.exit()
#for season in ['autumn', 'winter', 'growing']:#'spring', 'summer',
#for mo in range(12):
#for yr in range(2000,2023):
for season in ['all']:
    
    #season = 'year'
    #tif_name = 'dm_falshCount_{}.tif'.format(season)
    #tif_name = 'dm_falshCount_{}_{}.tif'.format(season,str(mo+1))
    tif_name = 'dm_flashCount_{}_{}.tif'.format(season,str(yr))
    tif_name2 = 'dm_flashDrought_{}_{}.tif'.format(season,str(yr))
    
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
        selectif = tif1[select,:,:] 
        select = dfts[(dfts.index.month == 9)|(dfts.index.month == 10)|(dfts.index.month == 11)]['ids']
    elif season == 'winter':
        select = dfts[(dfts.index.month == 12)|(dfts.index.month == 1)|(dfts.index.month == 2)]['ids']
        selectif = tif1[select,:,:] 
    elif season == 'growing':
        select = dfts[(dfts.index.month == 5)|(dfts.index.month == 6)|(dfts.index.month == 7)|(dfts.index.month == 8)|(dfts.index.month == 9)]['ids']
        selectif = tif1[select,:,:] 
    
    flashTif = np.empty((selectif[0].shape[0], selectif[0].shape[1]))
    flashTif[:] = np.nan
    
    flashTif2 = np.empty(selectif.shape)
    flashTif2[:] = np.nan
    
    
    
    for row in range(selectif[0].shape[0]):
        for col in range(selectif[0].shape[1]):
            dfcol = str(row)+'_'+str(col)
    
            ts = pd.Series(selectif[:,row, col], index = select.index, name = dfcol, dtype = np.int8).replace(15.0, np.nan)
            tsd = ts.diff(periods=period)
    
            conds = []
            for ix, difs in enumerate(tsd):
                if difs == 0:
                    val = 0   #0 = no change
                    conds.append(val)
                elif difs < 0:
                    val = 1   #1 = improvement
                    conds.append(val)
                elif difs == 1:
                    val = 2   #2 = deterioration
                    conds.append(val)
                elif difs > catchange:
                    if 3 in conds[ix-4:ix]:
                        val = 0 # flash drought already counted
                    else:
                        val = 3   #3 = flash drought
                    #val = 3
                    conds.append(val)
                else:
                    val = np.nan
                    conds.append(val)
                    
            #DF3[dfcol] = conds
            conds = pd.Series(conds, index = ts.index)
            
            count = len(conds[conds==3])
            flashTif[row,col] = count
            flashTif2[:,row,col] = conds
    
    #sys.exit()        
    with rio.open(outdir+tif_name, 'w', **tifmeta) as dst:
        dst.write(np.expand_dims(flashTif,0)) 
    #for the big tif with all data 
    tifmeta2 = tifmeta
    tifmeta2['count'] = flashTif2.shape[0]
    with rio.open(outdir+tif_name2, 'w', **tifmeta2) as dst:
        dst.write(np.expand_dims(flashTif2,0)) 

#a = dfts.shape[0]
#b = len(select)
#c = selectif.shape


'''

#Plots


import geopandas as gpd
from rasterstats import zonal_stats
states = gpd.read_file('/Users/ellenbw/Documents/UAH/SCO/NOAA_FD/Shae/SE_states.shp')
shp = '/Users/ellenbw/Documents/UAH/SCO/LGI/HUC12_SE/SEHuc12.shp'
gdf = gpd.read_file(shp)

tif = '../DM_FD_Cond1/dm_flashCount_all_.tif'

gdf['max'] = pd.DataFrame(
    zonal_stats(
        vectors=gdf['geometry'], 
        raster=tif, 
        stats='max'
    )
)['max']

fig, ax = plt.subplots() 

vmax = gdf['max'].max()

gdf.plot(column='max', cmap='hot_r',vmax = vmax,legend=True, ax = ax)
states.plot(facecolor = 'none', ax= ax)
ax.set_title('Cond1')


#poststamp plots
titles = ['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug', 'Sep','Oct','Nov','Dec']
titles = range(2000,2023)
states = gpd.read_file('SE_states.shp')
shp = '/Users/ellenbw/Documents/UAH/SCO/LGI/HUC12_SE/SEHuc12.shp'
gdf = gpd.read_file(shp)

#tifs = glob.glob('dm_falshCount_month*.tif').sort()
i = 0
fig, axs = plt.subplots(5,4)

for axv in axs:
    for axh in axv:
        
        tif = 'DM_FD_Cond1/dm_falshCount_month_{}.tif'.format(i+1)
    
        gdf['mean'] = pd.DataFrame(
            zonal_stats(
                vectors=gdf['geometry'], 
                raster=tif, 
                stats='mean'
            )
        )['mean']
        
        
       
        gdf.plot(column='mean', cmap='hot_r', vmax = 5,legend=False, ax = axh)
        states.plot(facecolor = 'none', ax= axh)
       
        axh.set_title(str(titles[i]))
        i+=1
        
sm = plt.cm.ScalarMappable(cmap='hot_r', norm=plt.Normalize(vmin=0, vmax=5))        
cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])    

fig.colorbar(sm, cax = cax)

'''



























