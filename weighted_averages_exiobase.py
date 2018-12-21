# load packages needed
import pandas as pd
import numpy as np
from osgeo import ogr
import shapely
from shapely.geometry import Point
import time
import scipy.stats as st
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr
import math
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
rev_and_int.to_csv('median_revenue_and_median_intensities_exiobase.csv', encoding='utf-8', index=True)

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

#AFTER THIS STEP THE WATER RISK RESKTORS CONTAIN 34 SECTORS
#safe
wss_water_intensities.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/Water_Risk_Sectors_intensities.csv', encoding='utf-8', index=True)

#compute range for each sector
#plotting done in excel
wss_range = wss_water_intensities.to_frame('water intensities').copy()
wss_range['range'] = sec_rev_int.groupby('Water Risk Sector')['water intensity (M3/EUR)'].agg(np.ptp)
wss_range['minimun'] = sec_rev_int.groupby('Water Risk Sector').min()['water intensity (M3/EUR)']
wss_range.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/WSS_Range.csv', encoding='utf-8', index=True)


#############################################################################
###########          WATER FOOTPRINTS                          ##############
#############################################################################


location_rev = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_locations_rev_fractions.csv', encoding='utf-8')
wss_water_intensities= pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/Water_Risk_Sectors_intensities.csv', encoding='utf-8', header = None)

wss_water_intensities = wss_water_intensities.set_index([0])
wss_water_intensities = pd.Series(wss_water_intensities.iloc[:,0])
water_intensities_dict = wss_water_intensities.to_dict()
water_intensities_dict = {k.lower(): v for k,v in water_intensities_dict.items()} # transform keys lo lower case

location_rev.columns = map(str.lower, location_rev.columns) # transform index to lower case

#define function
#calculates water footprint per location
water_footprints = location_rev.copy()

def compute_water_footprints (water_intensities_dict, location_rev):
    water_footprints = location_rev.copy()
    for key in water_intensities_dict:
        for column in location_rev:
            if key == column:
               water_footprints[column] = location_rev[column] * water_intensities_dict[key]


#run function
compute_water_footprints(water_intensities_dict, location_rev)

#save footprints
water_footprints.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_location.csv',
                        encoding='utf-8', index=False)

#sum up water footprint per company
water_footprints = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_location.csv',
                         encoding='utf-8')
MSCI_water_footprint_per_isin = water_footprints.drop(['longitude', 'latitude', 'number of locations', 'enterprise', 'bws', 'unnamed: 0', 'unnamed: 0', 'aggregated security name'], axis = 1).copy()
MSCI_water_footprint_per_isin = MSCI_water_footprint_per_isin.groupby(['name', 'isin']).sum()
MSCI_water_footprint_per_isin['global water footprint'] = MSCI_water_footprint_per_isin.iloc[:,2:].sum()
MSCI_water_footprint_per_isin = MSCI_water_footprint_per_isin.sum(1)
MSCI_water_footprint_per_isin = MSCI_water_footprint_per_isin.reset_index()

#MSCI_water_footprint_per_isin.index = MSCI_water_footprint_per_isin.index.get_level_values('isin')
#save
MSCI_water_footprint_per_isin.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_isin.csv',
                        encoding='utf-8', index=False)



#######################################################################################################################
#           directly calculate footprints per isin
#######################################################################################################################


total_revenue = pd.read_csv('export_rev_water_risk_big.csv', error_bad_lines=False, encoding='ISO-8859-1')
total_revenue.columns = map(str.lower, total_revenue.columns) # transform index to lower case

water_footprints_sectors = total_revenue.copy()
for key in water_intensities_dict:
    for column in total_revenue:
        if key == column:
            water_footprints_sectors[column] = total_revenue[column] * water_intensities_dict[key]

water_footprints_sectors = water_footprints_sectors.drop(['aggregated security name'], axis = 1).copy()

#MSCI_water_footprint_per_isin.index = MSCI_water_footprint_per_isin.index.get_level_values('isin')
#save
water_footprints_sectors.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_isin_and_sector.csv',
                        encoding='utf-8', index=False)

water_footprints_total = water_footprints_sectors.copy()
water_footprints_total['total footprint'] = water_footprints_total.loc[:,'bio power':'wind power'].sum(axis=1)
water_footprints_total = water_footprints_total.drop(water_footprints_total.columns[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]], axis=1)
water_footprints_total = water_footprints_total[(water_footprints_total[['total footprint']] != 0).all(axis=1)] #take out values where not revenue lead to no footprint
#save
water_footprints_total.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_isin_total.csv',
                        encoding='utf-8', index=False)

########################################RESULTS FOOTPRINTS##############################################################
water_footprints_sectors = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_isin_and_sector.csv',
                        encoding='utf-8',)

#AB InBev
inbev_rev = total_revenue.loc[total_revenue['ISIN'].isin(['BE0974293251'])]
inbev_rev = inbev_rev.drop(['ISIN', 'Name', 'Aggregated Security name'], axis = 1)
inbev_rev= inbev_rev.dropna(axis=1)

inbev_footprint = water_footprints_sectors.loc[water_footprints_sectors['isin'].isin(['BE0974293251'])]
inbev_footprint = inbev_footprint.drop(['isin', 'name'], axis = 1)
inbev_footprint= inbev_footprint.dropna(axis=1)

#revenue pie chart InBev
labels_inbev_rev = list(inbev_rev.columns.values)
labels_inbev_rev = ['food processing (60.00%)', 'office (10.00%)', 'retail (30.00%)']
labels_inbev_rev = [x.lower() for x in labels_inbev_rev]
fracs_inbev_rev = list(inbev_rev.iloc[0])
colors_inbev = ['#BDB76B', '#CD5C5C', '#5F9EA0']

#footprint pie chart InBev
labels_inbev_footprint = list(inbev_footprint.columns.values)
labels_inbev_footprint = ['food processing (98.82%)', 'office (0.33%)', 'retail (0.85%)']
fracs_inbev_footprint = list(inbev_footprint.iloc[0])

fig, axs = plt.subplots(1, 2)
axs[0].pie(fracs_inbev_rev, colors=colors_inbev, textprops=dict(fontsize=18), pctdistance=0.8)
axs[0].legend(labels_inbev_rev, loc=3, fontsize=18)
axs[1].pie(fracs_inbev_footprint, colors=colors_inbev, textprops=dict(fontsize=18), pctdistance=0.8)
axs[1].legend(labels_inbev_footprint, loc=3, fontsize=18)
axs[0].set_title('assigned revenue', horizontalalignment='center', verticalalignment='top', pad=50, fontweight= 'bold', fontsize=24)
axs[1].set_title('computed footprint', horizontalalignment='center', verticalalignment='top', pad=50, fontweight= 'bold', fontsize=24)
props = dict(boxstyle='round', facecolor='#FFFFF0', alpha=0.5)
plt.text(0.5, 0.5, 'total yearly footprint (computed): 190,514,567 m3', fontsize=18, horizontalalignment='right',verticalalignment='bottom', bbox=props)


#BAYER
bayer_rev = total_revenue.loc[total_revenue['ISIN'].isin(['DE000BAY0017'])]
bayer_rev = bayer_rev.drop(['ISIN', 'Name', 'Aggregated Security name'], axis = 1)
bayer_rev= bayer_rev.dropna(axis=1)

bayer_footprint = water_footprints_sectors.loc[water_footprints_sectors['isin'].isin(['DE000BAY0017'])]
bayer_footprint = bayer_footprint.drop(['isin', 'name'], axis = 1)
bayer_footprint= bayer_footprint.dropna(axis=1)

#revenue pie chart Bayer
labels_bayer_rev = list(bayer_rev.columns.values)
labels_bayer_rev = ['chemicals manufacture (28.49%)', 'office (62.53%)', 'other agriculture (2.22%)', 'other manufacture (6.75%)']
labels_bayer_rev = [x.lower() for x in labels_bayer_rev]
fracs_bayer_rev = list(bayer_rev.iloc[0])
colors_bayer = ['#DAA520', '#CD5C5C', '#6B8E23', '#C0C0C0']

#footprint pie chart Bayer
labels_bayer_footprint = list(bayer_footprint.columns.values)
labels_bayer_footprint = ['chemicals manufacture (88.60%)', 'office (2.37%)', 'other agriculture (3.10%)', 'other manufacture (5.92%)']
fracs_bayer_footprint = list(bayer_footprint.iloc[0])

fig, axs = plt.subplots(1, 2)
axs[0].pie(fracs_bayer_rev, colors=colors_bayer, textprops=dict(fontsize=18), pctdistance=0.8)
axs[0].legend(labels_bayer_rev, loc=3, fontsize=18)
axs[1].pie(fracs_bayer_footprint, colors=colors_bayer, textprops=dict(fontsize=18), pctdistance=0.8)
axs[1].legend(labels_bayer_footprint, loc=3, fontsize=18)
axs[0].set_title('assigned revenue', horizontalalignment='center', verticalalignment='top', pad=50, fontweight= 'bold', fontsize=24)
axs[1].set_title('computed footprint', horizontalalignment='center', verticalalignment='top', pad=50, fontweight= 'bold', fontsize=24)
props = dict(boxstyle='round', facecolor='#FFFFF0', alpha=0.5)
plt.text(0.5, 0.5, 'total yearly footprint (computed): 118,834,453 m3', fontsize=18, horizontalalignment='right',verticalalignment='bottom', bbox=props)


#show shares of contribution to a global water footprint
water_footprints_sectors = water_footprints_sectors.drop(['isin', 'name'], axis = 1)
global_footprint = water_footprints_sectors.sum()
global_footprint = global_footprint.sort_values()
global_footprint['others (24)'] = global_footprint[0:23].sum()
global_footprint = global_footprint[24:]

labels_global = list(global_footprint.index.values)
labels_global = ['other manufacture (1.43%)', 'metal manufacture(1.56%)', 'fertilizer (1.57%)', 'food processing (2.02%)', 'manufacture of machinery (2.27%)', 'chemicals manufacture (3.47%)', 'natural gas power (5.56%)', 'coal power (8.76%)', 'crop farming (23.88%)', 'nuclear power (45.31%)', 'others (24 sectors) (4.17%)']
fracs_global = list(global_footprint)
colors_global = ['#C0C0C0', '#FF8C00', '#20B2AA', '#BDB76B', '#778899', '#DAA520', '#DCDCDC', 'brown', '#2E8B57', 'yellowgreen', '#FFEBCD']

pie = plt.pie(fracs_global, pctdistance=0.8, colors=colors_global)
plt.legend(pie[0], labels=labels_global, bbox_to_anchor=(1, 0, 0.5, 1), loc="center left", fontsize=14)
plt.title('shares for MSCI water footprints', fontsize=20, fontweight= 'bold')
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)















#########################################DISKUSSION######################################################################
#calculate spearman rank correlation
water_footprints_total = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_water_footprint_per_isin_total.csv',
                         encoding='utf-8')
reuters_water_footprint = pd.read_csv('C:/Users/bod\Dropbox/1_Masterarbeit Carbon Delta/Themenfindung/Data/Water_Footprints/201808_HAP_WaterFootprints_MSCI_World.csv',
                         encoding='utf-8')

reuters_water_footprint = reuters_water_footprint.dropna(axis=0)
reuters_water_footprint = reuters_water_footprint.drop(['Name'], axis=1)
reuters_water_footprint = reuters_water_footprint.set_index('Enterprise ISIN')
reuters_dict = reuters_water_footprint.to_dict()
#map reuters footprint to calculated values
water_footprints_total['reuters footprints'] = water_footprints_total['isin'].map(reuters_dict['WaterWithdrawalTotal (cubic meters)'])

###################
#drop rows where Reuters has no reported data
water_footprints_total = water_footprints_total.dropna(subset=['reuters footprints'])
water_footprints_total = water_footprints_total.reset_index(drop=True)
##################

#spearman
spearmanr(water_footprints_total['total footprint'], water_footprints_total['reuters footprints'])

#pearson
pearsonr(water_footprints_total['total footprint'], water_footprints_total['reuters footprints'])

#standard deviation
stddev = math.sqrt(sum((water_footprints_total['total footprint'] - water_footprints_total['reuters footprints'])**2)/len(water_footprints_total))
stddev / water_footprints_total['total footprint'].mean()

#difference
water_footprints_total['difference'] = water_footprints_total['total footprint'] - water_footprints_total['reuters footprints']
sum(n < 0 for n in water_footprints_total['difference']) # total number of companies where reuters is larger
sum(n > 0 for n in water_footprints_total['difference']) # total number of companies where computed is larger

#difference as share of calculated footprints
water_footprints_total['rel_difference'] = water_footprints_total['difference'] / water_footprints_total['total footprint']
water_footprints_total['rel_difference'] = water_footprints_total['rel_difference'].abs() #make all values positive

#sort values
water_footprints_total = water_footprints_total.sort_values('rel_difference')

#difference form reuters
water_footprints_total['difference_from_reuters'] =  water_footprints_total['difference'] / water_footprints_total['reuters footprints']

#reuters / computed
water_footprints_total['reuters / computed'] = water_footprints_total['reuters footprints'] / water_footprints_total['total footprint']


#Histogram
#absolute difference
difference = water_footprints_total['difference'].tolist()
conf60 = st.t.interval(0.60, len(difference)-1, loc=np.mean(difference), scale=st.sem(difference))
#bins = np.linspace(conf60[0], conf60[1], num=100)
bins = np.linspace(min(difference), max(difference), num=100)
plt.hist(difference, bins, histtype='bar', rwidth=0.8)
plt.xlabel('difference', fontsize=26)
plt.ylabel('number of companies', fontsize=26)
plt.title('differences (computed - Reuters): all data', fontsize=28)
plt.show()

#relative differnece
#playing_with_intensities = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/Themenfindung/Data/Water intensities/playing_with_intensities.csv',
#                         encoding='utf-8')

rel_difference = water_footprints_total['reuters / computed'].tolist()
bins = np.linspace(0, 2, num=30)
plt.hist(rel_difference, bins, histtype='bar', rwidth=0.8)
plt.xlabel(' relative difference (Reuters/computed)', fontsize=26)
plt.ylabel('number of companies',fontsize=26)
plt.title('Reuters/computed', fontsize=28)
plt.show()

# difference from reuters
rel_difference = water_footprints_total['difference_from_reuters'].tolist()
bins = np.linspace(-1, 11, num=31)
plt.hist(rel_difference, bins, histtype='bar', rwidth=0.8)
plt.tick_params(labelsize=20, width=0.5)
plt.xlabel('(calculated - Reuters) / Reuters', fontsize=26)
plt.ylabel('number of companies',fontsize=26)
plt.title('Difference from Reuters', fontsize=28)
plt.show()



#extract companies with large differences
large_differences = water_footprints_total.copy()
large_differences['difference'] = water_footprints_total['difference'].abs()
large_differences = large_differences.loc[large_differences['difference'] > 10**10]
large_differences.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/large_differences.csv',
                        encoding='utf-8', index=False)

#remove large outliers and check quality measures
footprints_without_outliers = water_footprints_total[~water_footprints_total['isin'].isin(['DE0007037129', 'FR00102425511', 'IT0003128367', 'MU0117U00026', 'US25746U1097', 'US26441C2044','US30161N1019'])]
#pearson without outliers
spearmanr(footprints_without_outliers['total footprint'], footprints_without_outliers['reuters footprints'])
#spearman without outliers
pearsonr(footprints_without_outliers['total footprint'], footprints_without_outliers['reuters footprints'])


################################################################
#plot pie chart for RWE
################################################################
total_revenue = pd.read_csv('export_rev_water_risk_big.csv', error_bad_lines=False, encoding='ISO-8859-1')
rwe = total_revenue.loc[total_revenue['ISIN'].isin(['DE0007037129'])].reset_index()
rwe = rwe.drop(['ISIN', 'Name', 'Aggregated Security name', 'index'], axis = 1)
rwe = rwe.dropna(axis=1)

labels = list(rwe.columns.values)
labels = [x.lower() for x in labels]
fracs = list(rwe.iloc[0])
colors = ['green', 'brown', '#87CEFA', '#DCDCDC', 'yellowgreen', '#CD5C5C', '#708090', 'gold', 'olive']

#RWE: fracs for rwe are in TWh (Statista)
rwe_labels = ['coal', 'natural gas', 'nuclear', 'renewables', 'others']
rwe_fracs = [74.2+29.4,53.9, 30.3, 11.3, 3.1]
rwe_colors = ['brown', '#DCDCDC', 'yellowgreen', 'green', '#FFEBCD']


fig, axs = plt.subplots(1, 2)
axs[0].pie(fracs, labels=labels, autopct='%0.2f%%', explode=[0,.2,0,0,0,0,0,0,0], colors=colors, textprops=dict(fontsize=18), pctdistance=0.8)
axs[1].pie(rwe_fracs, labels=rwe_labels, autopct='%0.2f%%', explode=[.1,0,0,0,0], colors=rwe_colors,  textprops=dict(fontsize=18), pctdistance=0.8)
axs[0].set_title('assigned by model', horizontalalignment='center', verticalalignment='top', pad=50, fontweight= 'bold', fontsize=24)
axs[1].set_title('statista', horizontalalignment='center', verticalalignment='top', pad=50, fontweight= 'bold', fontsize=24)



###################################################################################
#plot pie chart for EDF
############################################################################
edf = total_revenue.loc[total_revenue['ISIN'].isin(['FR0010242511'])].reset_index()
edf = edf.drop(['ISIN', 'Name', 'Aggregated Security name', 'index', 'Geothermal Power'], axis = 1)# remove geothermal because its only .003% and not visible in chart
edf = edf.dropna(axis=1)

labels = list(edf.columns.values)
labels = [x.lower() for x in labels]
fracs = list(edf.iloc[0])
colors = ['green', 'brown', '#87CEFA', '#DCDCDC', 'yellowgreen', '#CD5C5C', '#708090', 'gold', 'olive']

#EDF: fracs for rwe are in TWh (homepage: https://www.edf.fr/mix-energetique)
edf_labels = ['other renewables', 'coal', 'hydro power', 'natural gas', 'nuclear', 'oil']
edf_fracs = [3,4,7,8,77,1]
edf_colors = ['green', 'brown', '#87CEFA' , '#DCDCDC', 'yellowgreen', '#FFEBCD']


fig, axs = plt.subplots(1, 2)
axs[0].pie(fracs, labels=labels, autopct='%0.2f%%', explode=[0,0,0,0,0.2,0,0,0,0], colors=colors, textprops=dict(fontsize=18), pctdistance=0.8)
axs[1].pie(edf_fracs, labels=edf_labels, autopct='%0.2f%%', explode=[0,0,0,0,0.2,0], colors=edf_colors, textprops=dict(fontsize=18), pctdistance=0.8)
axs[0].set_title('assigned by model', horizontalalignment='center', verticalalignment='top', pad=50, fontweight= 'bold', fontsize=24)
axs[1].set_title('EDF homepage', horizontalalignment='center', verticalalignment='top', pad=50, fontweight= 'bold', fontsize=24)



# looking on specific companies
neg_dif = total_revenue.loc[total_revenue['ISIN'].isin(['US30161N1019', 'FR0010242511', 'US26441C2044', 'US25746U1097'])]





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
###########          writing program to read location data    ##############
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



#############################################################################
###########          writing program to read location data     ##############
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




#############################################################################
###########                     Water VaRs                     ##############
#############################################################################

#get water intensiteis
wss_water_intensities = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/Water_Risk_Sectors_intensities.csv', encoding='utf-8', header=None, index_col=0)
#max_exposure is the table generated from the water intensities
wss_water_intensities.loc['hydro power'] = wss_water_intensities.loc['crop farming'] # set hydro to same intensity as crop farming to get 100 percent exposure

max_exposure = wss_water_intensities.copy()
max_exposure = pd.Series(max_exposure.iloc[:,0])

#for a linear relation between maximum exposure and intensity
for i in range(0, len(wss_water_intensities)):
    max_exposure.iloc[i] = wss_water_intensities.iloc[i,0] / max(wss_water_intensities[1])

del wss_water_intensities
###########################plot bar chart##############################
max_exposure_sort = max_exposure.sort_values(ascending=False)

heights = max_exposure_sort.values
bars = max_exposure_sort.index
y_pos = range(len(max_exposure_sort))
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.ylabel('relative maximum revenue lost', size=18)
plt.bar(y_pos, heights)
# Rotation of the bars names
plt.xticks(y_pos, bars, rotation=90)

######################plot damage curves for 3 different sectors##########################

plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)

m1 = max_exposure['crop farming'] / 0.6
m2 = max_exposure['nuclear power'] / 0.6
m3 = max_exposure['coal power'] / 0.6
m4 = max_exposure['natural gas power'] / 0.6

b1 =-0.4 * (max_exposure['crop farming']/0.6)
b2 =-0.4 * (max_exposure['nuclear power']/0.6)
b3 =-0.4 * (max_exposure['coal power']/0.6)
b4 =-0.4 * (max_exposure['natural gas power']/0.6)

plt.axis(ymin=0, xmax=1)
ax = plt.subplot(111)
ax.legend()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.plot(x, m1*x + b1, '-g', label='crop farming')# crop
plt.plot(x, m2*x + b2, '-b', label='nuclear power')# coal
plt.plot(x, m3*x + b3, '-r', label='coal power')# coal
plt.plot(x, m4*x + b4, '-y', label='natural gas power')# coal
plt.xlabel('BWS', size=14)
plt.ylabel('water VaR', size=14)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=12)

#######################plot linear damage function####################

#make a dict for max exposure
max_exposure = max_exposure.to_dict()

#link it with location revenue data
MSCI_locations_BWS_Projections = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_locations_BWS_Projections.csv', encoding='utf-8')
MSCI_locations_BWS_Projections.columns = map(str.lower, MSCI_locations_BWS_Projections.columns)
damage_per_loc = MSCI_locations_BWS_Projections.drop(['enterprise', 'unnamed: 0', 'aggregated security name'], axis = 1).copy()
del MSCI_locations_BWS_Projections

#try linear approach

def lin_dam_func(max_exp, bws):
    # if bws > 1:
    #     bws = 1
    # if bws < 0.4:
    #     bws = 0.4
    #bws = max(min(bws, 1),0.4)
    #bws = [v  if v<1 else 1 for v in bws]
    bws = np.array([max(min(v, 1),0.4) for v in bws]) #conditions in comprehension
    m = max_exp / 0.6
    b = -0.4 * (max_exp/0.6)
    rel_damage = bws * m + b
    return rel_damage



#####################Projections##################################

#################################################################
#################### BUSINESS AS USUAL SCENARIO #################
#################################################################

#compute VaR for every location with linear damage function für current BWS
for col in damage_per_loc.columns[4:38]:
    new_name = col+"_rev_exposed_cur"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws'])

#Business as usual 2020, linear damage funciton
for col in damage_per_loc.columns[4:38]:
    new_name = col + "_rev_exposed_bau20"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws 2020'])

#Business as usual 2030, linear damage funciton
for col in damage_per_loc.columns[4:38]:
    new_name = col + "_rev_exposed_bau30"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws 2030'])

# Business as usual 2040, linear damage funciton
for col in damage_per_loc.columns[4:38]:
    new_name = col + "_rev_exposed_bau40"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws 2040'])

damage_per_loc_bau = damage_per_loc.copy()
del damage_per_loc
damage_per_loc_bau.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_damage_per_loc_bau.csv', encoding='utf-8', index=True)

#sum var per location
damage_per_loc_bau['var_cur'] = damage_per_loc_bau.loc[:,'bio power_rev_exposed_cur':'wind power_rev_exposed_cur'].sum(1)
damage_per_loc_bau['var_20'] = damage_per_loc_bau.loc[:,'bio power_rev_exposed_bau20':'wind power_rev_exposed_bau20'].sum(1)
damage_per_loc_bau['var_30'] = damage_per_loc_bau.loc[:,'bio power_rev_exposed_bau30':'wind power_rev_exposed_bau30'].sum(1)
damage_per_loc_bau['var_40'] = damage_per_loc_bau.loc[:,'bio power_rev_exposed_bau40':'wind power_rev_exposed_bau40'].sum(1)

#delete sectordata and only leave sums per location
damage_per_loc_bau['revenue']=damage_per_loc_bau.loc[:,'bio power':'wind power'].sum(1) #calculate revenue per loc

#sum for all locations
var_per_isin_bau = damage_per_loc_bau[['name', 'isin', 'var_cur', 'var_20', 'var_30','var_40', 'revenue']].copy()
var_per_isin_bau = var_per_isin_bau.groupby(['name', 'isin']).sum()
var_per_isin_bau.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_var_per_isin_bau.csv', encoding='utf-8', index=True)

del damage_per_loc_pes


#################################################################
#################### PESSIMISTIC SCENARIO #######################
#################################################################

#current as basis
for col in damage_per_loc.columns[4:38]:
    new_name = col+"_rev_exposed_cur"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws'])

# pessimistic, linear damage funciton
for col in damage_per_loc.columns[4:38]:
    new_name = col + "_rev_exposed_pes20"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws 2020 pes'])

for col in damage_per_loc.columns[4:38]:
    new_name = col + "_rev_exposed_pes30"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws 2030 pes'])

for col in damage_per_loc.columns[4:38]:
    new_name = col + "_rev_exposed_pes40"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws 2040 pes'])

damage_per_loc_pes = damage_per_loc.copy()
del damage_per_loc

damage_per_loc_pes.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_damage_per_loc_pes.csv', encoding='utf-8', index=True)

#sum var per location
damage_per_loc_pes['var_cur'] = damage_per_loc_pes.loc[:,'bio power_rev_exposed_cur':'wind power_rev_exposed_cur'].sum(1)
damage_per_loc_pes['var_20'] = damage_per_loc_pes.loc[:,'bio power_rev_exposed_pes20':'wind power_rev_exposed_pes20'].sum(1)
damage_per_loc_pes['var_30'] = damage_per_loc_pes.loc[:,'bio power_rev_exposed_pes30':'wind power_rev_exposed_pes30'].sum(1)
damage_per_loc_pes['var_40'] = damage_per_loc_pes.loc[:,'bio power_rev_exposed_pes40':'wind power_rev_exposed_pes40'].sum(1)

#delete sectordata and only leave sums per location
damage_per_loc_pes['revenue']=damage_per_loc_pes.loc[:,'bio power':'wind power'].sum(1) #calculate revenue per loc

#sum for all locations
var_per_isin_pes = damage_per_loc_pes[['name', 'isin', 'var_cur', 'var_20', 'var_30','var_40', 'revenue']].copy()
var_per_isin_pes = var_per_isin_pes.groupby(['name', 'isin']).sum()
var_per_isin_pes.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_var_per_isin_pes.csv', encoding='utf-8', index=True)

del damage_per_loc_pes




#################################################################
#################### OPTIMISTIC SCENARIO ########################
#################################################################

#current as basis
for col in damage_per_loc.columns[4:38]:
    new_name = col+"_rev_exposed_cur"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws'])

#optimistic linear damage funciton
for col in damage_per_loc.columns[4:38]:
    new_name = col + "_rev_exposed_opt20"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws 2020 opt'])

for col in damage_per_loc.columns[4:38]:
    new_name = col + "_rev_exposed_opt30"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws 2030 opt'])

for col in damage_per_loc.columns[4:38]:
    new_name = col + "_rev_exposed_opt40"
    damage_per_loc[new_name] = damage_per_loc[col] * lin_dam_func(max_exp=max_exposure[col], bws=damage_per_loc['bws 2040 opt'])

damage_per_loc_opt = damage_per_loc.copy()
damage_per_loc_opt.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_damage_per_loc_opt.csv', encoding='utf-8', index=True)
del damage_per_loc

#sum var per location
damage_per_loc_opt['var_cur'] = damage_per_loc_opt.loc[:,'bio power_rev_exposed_cur':'wind power_rev_exposed_cur'].sum(1)
damage_per_loc_opt['var_20'] = damage_per_loc_opt.loc[:,'bio power_rev_exposed_opt20':'wind power_rev_exposed_opt20'].sum(1)
damage_per_loc_opt['var_30'] = damage_per_loc_opt.loc[:,'bio power_rev_exposed_opt30':'wind power_rev_exposed_opt30'].sum(1)
damage_per_loc_opt['var_40'] = damage_per_loc_opt.loc[:,'bio power_rev_exposed_opt40':'wind power_rev_exposed_opt40'].sum(1)

#delete sectordata and only leave sums per location
damage_per_loc_opt['revenue']=damage_per_loc_opt.loc[:,'bio power':'wind power'].sum(1) #calculate revenue per loc

#sum for all locations
var_per_isin_opt = damage_per_loc_opt[['name', 'isin', 'var_cur', 'var_20', 'var_30','var_40', 'revenue']].copy()
var_per_isin_opt = var_per_isin_opt.groupby(['name', 'isin']).sum()
var_per_isin_opt.to_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_var_per_isin_opt.csv', encoding='utf-8', index=True)


#####################################discussion and results VaR#########################################################

#plot shares of VaRs
damage_per_loc_bau = pd.read_csv('C:/Users/bod/Dropbox/1_Masterarbeit Carbon Delta/results/MSCI_damage_per_loc_bau.csv', encoding='ISO-8859-1', index_col=0)
var_per_isin_sectors_bau = damage_per_loc_bau.iloc[:,49:83] #select current scenario
var_per_isin_sectors_bau = var_per_isin_sectors_bau.sum()

var_per_isin_sectors_bau = var_per_isin_sectors_bau.sort_values()
var_per_isin_sectors_bau['others (24)'] = var_per_isin_sectors_bau[0:23].sum()
var_per_isin_sectors_bau = var_per_isin_sectors_bau[24:]

labels_global = list(var_per_isin_sectors_bau.index.values)
labels_global = ['metal manufacture (1.32%)', 'other manufacture (1.36%)', 'food processing (1.86%)', 'manufacture of machinery (2.12%)', 'chemicals manufacture (2.77%)', 'natural gas power (3.57%)', 'coal power (5.12%)', 'hydro power (21.28%)', 'nuclear power (24.37%)', 'crop farming (32.06%)', 'others (24 sectors) (4.17%)']
fracs_global = list(var_per_isin_sectors_bau)
colors_global = ['#FF8C00', '#C0C0C0', '#BDB76B', '#778899', '#DAA520', '#DCDCDC', 'brown', '#87CEFA', 'yellowgreen', '#2E8B57', '#FFEBCD']

pie = plt.pie(fracs_global, pctdistance=0.8, colors=colors_global)
plt.legend(pie[0], labels=labels_global, bbox_to_anchor=(1, 0, 0.5, 1), loc="center left", fontsize=14)
plt.title('shares in summed VaR (current conditions)', fontsize=20, fontweight= 'bold')
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)



########################################


    # s-shape funktion to calculate exposure factor
    # copied from Sam
    # Vhalf is the the point of the function where half of the damage possible is caused (Wendepunkt) here between 0.4 and 1.0 -> 0.7
    # vThreshold is the threshold below which no damage ocuurs (here from 0.4 on)
    # once wri exceeds vThreshold, vTemp gets positive
    # what is scale?

    scale = 1.0,
    vHalf = 0.7
    vThreshold = 0.4
    wri = 0.5
    for vTemp = ((wri - vThreshold) / (vHalf - vThreshold)):
        if vTemp < 0:  # is that effective?
            vTemp = 0
        exp = scale * vTemp ** 3 / (1 + vTemp ** 3)
        return exp


    def createImpactFuncEmanuel(scale=1.0, vHalf=0.7, vThreshold=0.4, wri=0):
        vTemp = ((wri - vThreshold) / (vHalf - vThreshold))
        if vTemp < 0:  # is that effective?
            vTemp = 0
        exp = scale * vTemp ** 3 / (1 + vTemp ** 3)
        return exp


    # a simple division by the total number of all locations isnt possiblie since




    # check time of a process############################################################################################
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
