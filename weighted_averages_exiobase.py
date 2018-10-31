# load packages
import pandas as pd
import codecs
import numpy as np
from osgeo import ogr
import shapely
from shapely.geometry import Point
import time

#loading data
#A
# A shows how much a sector buys from another to create one euro of rev (value purchased / value rev)
# deleted this comment
doc = codecs.open('A.txt','rU')
A = pd.read_csv(doc, sep='\t')
A.rename(columns=A.iloc[0,:],inplace=True)
del A['sector']
A.drop(A.index[0])
A = A.iloc[2:,0:]
A = A.reset_index(drop=True)
A.iloc[:,1:] = A.iloc[:,1:].astype(float)
sumA = A.iloc[:,1:].sum().reset_index()


#F
#
doc = codecs.open('F.txt','rU')
F = pd.read_csv(doc, sep='\t')
F.rename(columns=F.iloc[0,:],inplace=True)

#extracting added value (AV) from table F
AV = F.iloc[1:10,:]
del AV['sector']
AV.iloc[0:,0:]=AV.iloc[0:,0:].astype(float)
AV = AV.sum().reset_index() # results in a 7987 rows df (49 countries x 163 industries)

###############################################################################
#calculating Total revenue per sector
###############################################################################

rev = sumA.copy().reset_index(drop=True)
#Create table
rev_matrix = pd.DataFrame(np.nan, index=range(0,163), columns=range(0,49))
rev_matrix.iloc[:,0] = rev.iloc[:,0]


#fill table with values
# revenue = AV /(1-A)
for i in range (0,48):
    for k in range (0,163):
        rev_matrix.iloc[k,i+1] = AV.iloc[k+(163*i),1] / (1-sumA.iloc[k+(163*i),1])

#exclude zero values
rev_matrix = rev_matrix.replace(0, np.nan)

#median and mean over all columns, should take median; mean ist just calculate for comparing
# unit is M.EUR since AV is M.EUR and A is M.EUR/M.EUR
rev_median = rev_matrix.iloc[:,1:].median(axis = 1, skipna=True)
rev_mean = rev_matrix.iloc[:,1:].mean(axis = 1, skipna=True)

#save files / yet to run
rev_median.to_csv('revenue_sectoral_median_exiobase.csv', encoding='utf-8', index=False)
rev_mean.to_csv('revenue_sectoral_mean_exiobase.csv', encoding='utf-8', index=False)

##################################################################################################
#the exiobase sector averages were assigend a Sector by hand in EXCEL
#in the first step, the sectors were CD sector system plus some additional
#the wighted average is calculated using the median revenue of a sector
###############################################################################################

sec_rev_int = pd.read_csv('EXIOBASE_rev_and_intensities_with_sectorcode.csv', error_bad_lines=False, encoding='ISO-8859-1')
weighted_avg = sec_rev_int.groupby(['Sector Code']).apply(lambda x: np.average(x['m3/EUR'], weights=x['median revenue in M.EUR']))


##################################################################
#write subroutine to create damage function
#first step: function to match the intensity with a maximum exposure value
#second step
######################################################

#max_exposure is the table generated from the water intensities
max_exposure = weighted_avg.copy()
#for a linear relation between maximum exposure and intensity
for i in range(0,weighted_avg.shape[0]):
    max_exposure[i] = weighted_avg[i] / max(weighted_avg)


# s-shape funktion to calculate exposure factor
#copied from Sam
#Vhalf is the the point of the function where half of the damage possible is caused (Wendepunkt) here between 0.4 and 1.0 -> 0.7
#vThreshold is the threshold below which no damage ocuurs (here from 0.4 on)
#once wri exceeds vThreshold, vTemp gets positive
#what is scale?

def createImpactFuncEmanuel(scale=1.0, vHalf=0.7, vThreshold=0.4, wri=0):
    vTemp = ((wri - vThreshold) / (vHalf - vThreshold))
    if vTemp < 0: # is that effective?
        vTemp = 0
    exp = scale * vTemp ** 3 / (1 + vTemp ** 3)
    return exp


#load location and sectoral revenue database
company_data = pd.read_csv('locations_rev_sector_msci.csv', error_bad_lines=False, encoding='ISO-8859-1')

#load shapefile with layer BWS (package shapefile)
sf = ogr.Open("aqueduct_global_dl_20150409.shp")
layer = sf.GetLayerByName("aqueduct_global_dl_20150409")
spatialref = layer.GetSpatialRef()

company_data1 = company_data.iloc[:10,:].copy()
company_data1['BWS'] = np.nan
# Location for Pizzeria Gusto Italy: LAT:49.873626 N, LONG:-97.183495 E (Canada)
# woori bank china: 22.5347431 N, 114.0220717 E
# note: syntax for POINT is POINT(LONG LAT)

for i in range(0, 10):
    long = company_data.loc[i, 'longitude']
    lat = company_data.loc[i, 'latitude']
    point = shapely.geometry.Point(long, lat)
    point = ogr.CreateGeometryFromWkt(str(point))
    for feature in layer:
        if feature.GetGeometryRef().Contains(point):
            company_data1.loc[i,'BWS'] = feature.GetField("BWS")
            break
    layer.ResetReading()

#GetField gets the data of a column. Here 9 is for BWS

# imp_fun.plot();

#check time of a process############################################################################################
start = time.time()
long = company_data.loc[10, 'longitude']
lat = company_data.loc[10, 'latitude']
point = shapely.geometry.Point(long, lat)
point = ogr.CreateGeometryFromWkt(str(point))
for feature in layer:
    if feature.GetGeometryRef().Contains(point):
        BWS = feature.GetField("BWS")
        print(feature.GetField("BWS"))
        break
layer.ResetReading()

end = time.time()
print(end - start)
####################################################################################################################



min_max_scaler = preprocessing.MinMaxScaler()

max_exposure = weighted_avg

max(weighted_avg)

weighted_avg.count(cow)