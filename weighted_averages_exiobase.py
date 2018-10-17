# load packages
import pandas as pd
import codecs
import numpy as np

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
#calculating Total Output
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
