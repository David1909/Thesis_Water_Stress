# load packages needed
import pandas as pd
import codecs
import numpy as np
from osgeo import ogr
import shapely
from shapely.geometry import Point
import time
import math
import scipy.stats as st
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

#############################################################################
###########          computing average water intensities        #############
#############################################################################

#read consumption and withdrawal data
withdrawals = pd.read_csv('exiobase_F_total_withdrawals.csv', error_bad_lines=False, encoding='ISO-8859-1')
consumption = pd.read_csv('exiobase_F_total_consumption.csv', error_bad_lines=False, encoding='ISO-8859-1')

withdrawals = withdrawals.drop([0])
withdrawals.rename(columns=withdrawals.iloc[0], inplace=True)
withdrawals = withdrawals.drop([1])
withdrawals = withdrawals.set_index('sector', drop=True)
withdrawals = withdrawals.astype('float')


consumption = consumption.drop([0])
consumption.rename(columns=consumption.iloc[0], inplace=True)
consumption = consumption.drop([1])
consumption = consumption.set_index('sector', drop=True)
consumption = consumption.astype('float')

#sum water intensities for every country and every sector
sum_withdrawals = withdrawals.sum(0)
sum_consumption = consumption.sum(0)

#replace withdrawals if smaller than consumption
#combined_df = sum_withdrawals.copy()
#for i in range(0,len(combined_df)):
#    if sum_consumption.iat[i] > combined_df.iat[i]:
#        combined_df.iat[i] = sum_consumption.iat[i]

#replace zeros with NaN
combined_df = sum_consumption + sum_withdrawals
combined_df = combined_df.replace(0, np.nan)

#create a table with coutry x sector
# cutting the Series in 49 rows and 163 columns
water_use = pd.DataFrame(np.nan, index=range(0,163), columns=range(0,49))
water_use.index = combined_df.index[0:163]

for i in range(0,49):
    for k in range(0,163):
        water_use.iloc[k,i] = combined_df.iat[k + (163 * i)]

#calculate sectoral water intensities by taking the median or mean for every sector
median_sector_water_intensities = new_df.median(skipna=True)
mean_sector_water_intensities = new_df.mean(skipna=True)



#############################################################################
###########          computing weights for weighted averages    #############
#############################################################################

A = pd.read_fwf('A.txt')


#loading economic data
#A
# A shows how much a sector buys from another to create one euro of rev (value purchased / value rev)
# deleted this comment
sumA = pd.read_csv('sumA.csv', sep="\t", error_bad_lines=False, encoding='ISO-8859-1', header=1, index_col=0)
sumA = sumA.T #transpose

#F
#
F = pd.read_csv('F.txt', sep="\t", header=None, low_memory=False)
F = F.drop([0])
F.rename(columns=F.iloc[0], inplace=True)
F = F.drop([1])
F.reset_index(inplace=True, drop=True)

#extracting added value (AV) from table F
AV = F.iloc[0:8,:]
del AV['sector']
AV=AV.astype(float)
AV = AV.sum().reset_index() # results in a 7987 rows df (49 countries x 163 industries)
AV = AV.set_index('index', drop=True)

#calculating Total revenue per sector

rev = sumA.copy()
#Create table
rev_matrix = pd.DataFrame(np.nan, index=range(0,163), columns=range(0,49))
rev_matrix.index = rev.index[0:163]


#fill table with values
# revenue = AV /(1-A)
for i in range (0,49):
    for k in range (0,163):
        rev_matrix.iloc[k,i] = AV.iloc[k+(163*i),0] / (1-sumA.iloc[k+(163*i),0])

#exclude zero values
rev_matrix = rev_matrix.replace(0, np.nan)

#median and mean over all columns, should take median; mean ist just calculate for comparing
# unit is M.EUR since AV is M.EUR and A is M.EUR/M.EUR
rev_median = rev_matrix.iloc[:,1:].median(axis = 1, skipna=True)
rev_mean = rev_matrix.iloc[:,1:].mean(axis = 1, skipna=True)

#save files / yet to run
rev_median.to_csv('revenue_sectoral_median_exiobase.csv', encoding='utf-8', index=False)
rev_mean.to_csv('revenue_sectoral_mean_exiobase.csv', encoding='utf-8', index=False)


#############################################################################
###########          combine revenue and water intensities    ###############
#############################################################################


ex_3_water_int = water_use / rev_matrix
ex_3_water_int = ex_3_water_int.median(axis = 1, skipna=True)

rev_and_int = pd.DataFrame(data=median_sector_water_intensities)
rev_and_int = rev_and_int.assign(revenue=rev_median.values)
#print csv:
rev_and_int.to_csv('median_revenue_and_median_intensities_exiobase.csv', encoding='utf-8', index=False)

##################################################################################################
########      the exiobase sector averages were assigend a Sector by hand in EXCEL      ##########
########      creation of sector by simalarity of water intensity and Business Model    ##########
########      the weighted average is calculated using the median revenue of a sector   ##########
##################################################################################################

sec_rev_int = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/Themenfindung/Data/Mapping/Mapping_median_revenue_and_median_intensities_exiobase.csv', error_bad_lines=False, encoding='ISO-8859-1')
sec_rev_int['Water Risk Sector'] = sec_rev_int['Water Risk Sector'].str.lower()
sec_rev_int = sec_rev_int.set_index('EXIOBASE Sector')
sec_rev_int = sec_rev_int.copy().drop(index='Private households with employed persons')
wss_water_intensities = sec_rev_int.groupby(['Water Risk Sector']).apply(lambda x: np.average(x['water intensity (M3/EUR)'], weights=x['revenue (M.EUR)']))
wss_water_intensities = wss_water_intensities /1.13 # transform to dollar with 1.13 exchange rate (average 2017)

#compute range for each sector
#plotting done in excel
wss_range = wss_water_intensities.to_frame('water intensities').copy()
wss_range['range'] = sec_rev_int.groupby('Water Risk Sector')['water intensity (M3/EUR)'].agg(np.ptp)
wss_range['minimun'] = sec_rev_int.groupby('Water Risk Sector').min()['water intensity (M3/EUR)']
wss_range.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/WSS_Range.csv', encoding='utf-8', index=True)

#
#AFTER THIS STEP THE WATER RISK RESKTORS CONTAIN 34 SECTORS
#safe
wss_water_intensities.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/Water_Risk_Sectors_intensities.csv', encoding='utf-8', index=True)


#############################################################################
###########         merge location data and revenue                   #######
###########    result: company locations with fraction of revenue     #######
#############################################################################

locations = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_locations_BWS.csv', error_bad_lines=False, encoding='ISO-8859-1')
locations=locations.drop(locations.columns[[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]], axis=1).copy()

total_revenue = pd.read_csv('export_rev_water_risk_big.csv', error_bad_lines=False, encoding='ISO-8859-1')
location_rev = locations.merge(total_revenue, on='ISIN')

# get number of locations for dividing revenue by number of locations
location_number = location_rev.groupby('ISIN').count()
location_number = location_number['latitude']

#create dictionary for ISINs and map dictionary to the df as new column
isin_dict = location_number.to_dict()
location_rev['number of locations'] = location_rev['ISIN'].map(isin_dict)

#loop divides revenues by number of locations
for i in range(0,len(location_rev)):
    for k in range(7, (len(location_rev.columns)-1)):
        location_rev.iloc[i,k] = location_rev.iloc[i,k] / location_rev.iloc[i,41]

# check if number of locations in ols dataframe (phanos) equals the count in location_rev
#locations.loc[locations['ISIN'] == 'TW0002610003']

location_rev.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_locations_rev_fractions.csv', encoding='utf-8', index=False)


#############################################################################
###########          writing programm to read location data    ##############
###########                 CURRENT DATA, not projection data  ##############
#############################################################################

#load location and sectoral revenue database
company_data = pd.read_csv('locations_rev_sector_msci.csv', error_bad_lines=False, encoding='ISO-8859-1')

#load shapefile with layer BWS (package shapefile)
sf = ogr.Open("aqueduct_global_dl_20150409.shp")
layer = sf.GetLayerByName("aqueduct_global_dl_20150409")
spatialref = layer.GetSpatialRef()

company_data1 = company_data.copy()
company_data1['BWS'] = np.nan
# Location for Pizzeria Gusto Italy: LAT:49.873626 N, LONG:-97.183495 E (Canada)
# woori bank china: 22.5347431 N, 114.0220717 E
# note: syntax for POINT is POINT(LONG LAT)

for i in range(0, len(company_data['latitude'])):
    long = company_data.loc[i, 'longitude']
    lat = company_data.loc[i, 'latitude']
    point = shapely.geometry.Point(long, lat)
    point = ogr.CreateGeometryFromWkt(str(point))
    for feature in layer:
        if feature.GetGeometryRef().Contains(point):
            company_data1.loc[i,'BWS'] = feature.GetField("BWS")
            break
    layer.ResetReading()

company_data1.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_locations_BWS.csv', encoding='utf-8', index=False)

#check the percentage of outliers (nan values)
location_rev['BWS'].isna().sum()/len(location_rev['latitude'])
#error is at 0.04949630816782096
#GetField gets the data of a column. Here 9 is for BWS




#############################################################################
###########          WATER FOOTPRINTS                          ##############
#############################################################################


#location_rev = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_locations_rev_fractions.csv', encoding='utf-8')
wss_water_intensities= pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/Water_Risk_Sectors_intensities.csv', encoding='utf-8', header = None)

wss_water_intensities = wss_water_intensities.set_index([0])
wss_water_intensities = pd.Series(wss_water_intensities.iloc[:,0])
water_intensities_dict = wss_water_intensities.to_dict()
water_intensities_dict = {k.lower(): v for k,v in water_intensities_dict.items()} # transform keys lo lower case

location_rev.columns = map(str.lower, location_rev.columns) # transform index to lower case

#define function
#calculates water footprint per location

def compute_water_footprints (water_intensities_dict, location_rev):
    water_footprints = location_rev.copy()
    for key in water_intensities_dict:
        for column in location_rev:
            if key == column:
               water_footprints[column] = location_rev[column] * water_intensities_dict[key]


#run function
compute_water_footprints (water_intensities_dict, location_rev)

#save footprints
water_footprints.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_location.csv',
                        encoding='utf-8', index=True)

#sum up water footprint per company
water_footprints = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_location.csv',
                         encoding='utf-8')
MSCI_water_footprint_per_isin = water_footprints.drop(['longitude', 'latitude', 'number of locations', 'enterprise', 'bws', 'unnamed: 0', 'Unnamed: 0', 'aggregated security name'], axis = 1).copy()
MSCI_water_footprint_per_isin = MSCI_water_footprint_per_isin.groupby(['name', 'isin']).sum()
MSCI_water_footprint_per_isin['global water footprint'] = MSCI_water_footprint_per_isin.iloc[:,2:].sum()
MSCI_water_footprint_per_isin = MSCI_water_footprint_per_isin.sum(1)
MSCI_water_footprint_per_isin = MSCI_water_footprint_per_isin.reset_index()

#MSCI_water_footprint_per_isin.index = MSCI_water_footprint_per_isin.index.get_level_values('isin')
#save
MSCI_water_footprint_per_isin.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_isin.csv',
                        encoding='utf-8', index=True)




#######################################################################################################################
#new approach
#######################################################################################################################

total_revenue.columns = map(str.lower, total_revenue.columns) # transform index to lower case


water_footprints_sectors = total_revenue.copy()
for key in water_intensities_dict:
    for column in total_revenue:
        if key == column:
            water_footprints_sectors[column] = total_revenue[column] * water_intensities_dict[key]

water_footprints_sectors = water_footprints_sectors.drop(['aggregated security name'], axis = 1).copy()
water_footprints_sectors = water_footprints_sectors.iloc[0:].sum(1)

#MSCI_water_footprint_per_isin.index = MSCI_water_footprint_per_isin.index.get_level_values('isin')
#save
water_footprints_sectors.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_isin_2.csv',
                        encoding='utf-8', index=True)







#########################################DISKUSSION######################################################################
#calculate spearman rank correlation
MSCI_water_footprint_per_isin = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_isin.csv',
                         encoding='utf-8', header = None)
reuters_water_footprint = pd.read_csv('C:/Users/bod\Dropbox/1_Masterarbeit Carbon Delta/Themenfindung/Data/Water_Footprints/201808_HAP_WaterFootprints_MSCI_World.csv',
                         encoding='utf-8')

reuters_water_footprint = reuters_water_footprint.dropna(axis=0)
reuters_water_footprint = reuters_water_footprint.drop(['Name'], axis=1)
reuters_water_footprint = reuters_water_footprint.set_index('Enterprise ISIN')
reuters_dict = reuters_water_footprint.to_dict()
#map reuters footprint to calculated values
MSCI_water_footprint_per_isin['reuters footprints'] = MSCI_water_footprint_per_isin[1].map(reuters_dict['WaterWithdrawalTotal (cubic meters)'])

###################
#drop rows where Reuters has not reported data
MSCI_water_footprint_per_isin = MSCI_water_footprint_per_isin.dropna(subset=['reuters footprints'])
MSCI_water_footprint_per_isin = MSCI_water_footprint_per_isin.reset_index(drop=True)
##################

#spearman
spearmanr(MSCI_water_footprint_per_isin.iloc[:,2], MSCI_water_footprint_per_isin['reuters footprints'])

#pearson
pearsonr(MSCI_water_footprint_per_isin.iloc[:,2], MSCI_water_footprint_per_isin['reuters footprints'])

#standard deviation
stddev = math.sqrt(sum((MSCI_water_footprint_per_isin.iloc[:,2] - MSCI_water_footprint_per_isin['reuters footprints'])**2)/len(MSCI_water_footprint_per_isin))

#Histogram
#absolute difference
difference = MSCI_water_footprint_per_isin['difference'].tolist()
conf60 = st.t.interval(0.60, len(difference)-1, loc=np.mean(difference), scale=st.sem(difference))
bins = np.linspace(conf60[0], conf60[1], num=100)
# or bins = np.linspace(min(difference), max(difference), num=100)
plt.hist(difference, bins, histtype='bar', rwidth=0.8)
plt.xlabel('difference', )
plt.ylabel('number of companies')
plt.title('differences (computed - Reuters): all data')
plt.show()

#relative differnece
playing_with_intensities = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/Themenfindung/Data/Water intensities/playing_with_intensities.csv',
                         encoding='utf-8')
rel_difference = playing_with_intensities['Reuters / sum calculated'].tolist()
bins = np.linspace(0, 2, num=30)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)
plt.hist(rel_difference, bins, histtype='bar', rwidth=0.8)
plt.xlabel(' relative difference (Reuters/computed)')
plt.ylabel('number of companies')
plt.title('Reuters/computed')
plt.show()



#looking on specific companies
neg_dif = total_revenue.loc[total_revenue['ISIN'].isin(['US30161N1019', 'FR0010242511', 'US26441C2044', 'US25746U1097'])]










#############################################################################
###########          writing programm to read location data    ##############
###########                 future projections                 ##############
#############################################################################

#load shapefile with layer BWS (package shapefile)
sf = ogr.Open("C:/Users/bod/Desktop/aqueduct_projections_20150309_shp/aqueduct_projections_20150309.shp")
layer = sf.GetLayerByName("aqueduct_projections_20150309")
spatialref = layer.GetSpatialRef()

#loading data
MSCI_locations = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_locations_rev_fractions.csv', encoding='utf-8')

#BWS for business as usual
MSCI_locations['BWS 2020'] = np.nan
MSCI_locations['BWS 2030'] = np.nan
MSCI_locations['BWS 2040'] = np.nan

#rows for pessimistic scenarios
MSCI_locations['BWS 2020 pes'] = np.nan
MSCI_locations['BWS 2030 pes'] = np.nan
MSCI_locations['BWS 2040 pes'] = np.nan

#rows for optimistic scenario
MSCI_locations['BWS 2020 opt'] = np.nan
MSCI_locations['BWS 2030 opt'] = np.nan
MSCI_locations['BWS 2040 opt'] = np.nan

#for loop extracting water stress values for 2020, 2030, 2040
for i in range(0, len(MSCI_locations['latitude'])):
    long = MSCI_locations.loc[i, 'longitude']
    lat = MSCI_locations.loc[i, 'latitude']
    point = shapely.geometry.Point(long, lat)
    point = ogr.CreateGeometryFromWkt(str(point))
    for feature in layer:
        if feature.GetGeometryRef().Contains(point):
            MSCI_locations.loc[i, 'BWS 2020'] = feature.GetField("ws2028tr")
            MSCI_locations.loc[i, 'BWS 2030'] = feature.GetField("ws3028tr")
            MSCI_locations.loc[i, 'BWS 2040'] = feature.GetField("ws4028tr")
            MSCI_locations.loc[i, 'BWS 2020 pes'] = feature.GetField("ws2038tr")
            MSCI_locations.loc[i, 'BWS 2030 pes'] = feature.GetField("ws3038tr")
            MSCI_locations.loc[i, 'BWS 2040 pes'] = feature.GetField("ws4038tr")
            MSCI_locations.loc[i, 'BWS 2020 opt'] = feature.GetField("ws2024tr")
            MSCI_locations.loc[i, 'BWS 2030 opt'] = feature.GetField("ws3024tr")
            MSCI_locations.loc[i, 'BWS 2040 opt'] = feature.GetField("ws4024tr")
            break
    layer.ResetReading()
MSCI_locations.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_locations_BWS_Projections.csv', encoding='utf-8', index=False)




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


start = time.time()
long = MSCI_locations.loc[10, 'longitude']
lat = MSCI_locations.loc[10, 'latitude']
point = shapely.geometry.Point(long, lat)
point = ogr.CreateGeometryFromWkt(str(point))
for feature in layer:
    if feature.GetGeometryRef().Contains(point):
        MSCI_locations.loc[10, 'BWS 2020'] = feature.GetField("ws2028tr")
        MSCI_locations.loc[10, 'BWS 2030'] = feature.GetField("ws3028tr")
        MSCI_locations.loc[10, 'BWS 2040'] = feature.GetField("ws4028tr")
        MSCI_locations.loc[10, 'BWS 2020 pes'] = feature.GetField("ws2038tr")
        MSCI_locations.loc[10, 'BWS 2030 pes'] = feature.GetField("ws3038tr")
        MSCI_locations.loc[10, 'BWS 2040 pes'] = feature.GetField("ws4038tr")
        MSCI_locations.loc[10, 'BWS 2020 opt'] = feature.GetField("ws2024tr")
        MSCI_locations.loc[10, 'BWS 2030 opt'] = feature.GetField("ws3024tr")
        MSCI_locations.loc[10, 'BWS 2040 opt'] = feature.GetField("ws4024tr")
        break
layer.ResetReading()

end = time.time()
print(end - start)


#############################################################################
###########                     Water VaRs                     ##############
#############################################################################

#get water intensiteis
wss_water_intensities = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/Water_Risk_Sectors_intensities.csv', encoding='utf-8', header=None)


#max_exposure is the table generated from the water intensities
max_exposure = wss_water_intensities.copy()

#for a linear relation between maximum exposure and intensity
for i in range(0,wss_water_intensities.shape[0]):
    max_exposure[i] = wss_water_intensities[i] / max(wss_water_intensities)


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
#a simple division by the total number of all locations isnt possiblie since