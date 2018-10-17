#Processing Reuters Sample


###########################################
#########largest sample ###################
###########################################

#loading packages
import pandas as pd

sample = pd.read_csv('water_intensities_larger_sample.csv', error_bad_lines=False, encoding='ISO-8859-1')
sample = sample.dropna()

#compute sector averages
sample_sec_average = sample.groupby('Main_Sector').median().reset_index()
sample_sec_average.to_csv('water_intensities_larger_sample_sec_median.csv', encoding='utf-8', index=False)

#print CSV
MSCI_sec_average.to_csv('MSCI_sec_intensity_median.csv', encoding='utf-8', index=False)
MSCI_subsec_average.to_csv('MSCI_subsec_median.csv', encoding='utf-8', index=False)

##################################
########     MSCI    #############
##################################


#import of MSCI-CSV
MSCI = pd.read_csv('201808_HAP_WaterUsetoRev_MSCI_World.csv', delimiter=';', decimal=',', error_bad_lines=False)

#delete rows with NaN values
MSCI_sorted = MSCI.dropna()
print(MSCI_sorted)

# turn water intensity column into numeric
MSCI_sorted['Water_Use_per_revenue_TRAnalyticWaterUse_m3perUSD'] = pd.to_numeric(MSCI_sorted['Water_Use_per_revenue_TRAnalyticWaterUse_m3perUSD'], errors='coerce')
MSCI_sorted.dtypes #check format for each column#
MSCI_sorted = MSCI_sorted.groupby('Main_Sector').median().reset_index()

#print CSV
MSCI_sorted.to_csv('MSCI_sec_median.csv', encoding='utf-8', index=False)