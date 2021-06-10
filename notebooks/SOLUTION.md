Our Solution Can be divided into two Approaches :

##First Approach :

The first Approach can be divided into 4 parts :

###1st Part :  Vegetation Index Data Creation
####Step 1 :

The Vegetation dataset index is created from some statistics over : NDVI , GRNDVI , EVI , SAVI , CCCI . statistics used are :

MEDIAN ( recommended by experts )  [ NDVI , GRNDVI , EVI ]
MAX  [NDVI , GRNDVI , EVI , SAVI , CCCI ]
MIN [NDVI , GRNDVI ]
####Step 2 : 
Referred to some visualization  we discover that NDVI, SAVI, GRNDVI have the same distribution for specific months  . so we created a function that create over those months :

Products of NDVI, SAVI, GRNDVI features
std of NDVI, SAVI, GRNDVI features
mean of NDVI, SAVI, GRNDVI features
####Step 3 : 
apply Yeo-Johnson transformation to transform data distribution to a NORMAL Distribution .

###2nd part :  Transform the Additional DATA
After doing some research and referring to some experts, we found that :

Maize Season in Kenya is from mars to October.
Precipitation, Minimum temperature, Maximum temperature.
soil features are very useful.
so we created a function that transforms the Additional data by Calculating the mean over 4 years from 3rd 3 to month 10. For example :

we take month 3 and then we create a feature average_per_4Years_on_month_3 which is the mean over [ month_3_2016,month_3_2017,month_3_2018,month_3_2019 ] and like that ....

###3rd part : Create Red Bands DataSet
We created features from statistics over relation between Red Bands (this is also recommended by some experts in this field) For example, we calculate for month 5 :

####step 1 : b7_b6_array = 5_S2_B7 / 5_S2_B6
####step 2 : we calculate the median over the resulting array
And Finally, we concatenated those Datasets to get a 233 features dataset.

Btw in this approach, we're using only quality 1 and 3, adding quality 2 improves CV but makes LB very Bad.

###4th part : Modeling
Using  5 Kfold splits with shuffle =True.
Wokring with Xgboost with colsample_bytree = 0.65 .
in this Approach Our cv is 1.59 - LB is 1.65.

##Second Approach :
The second approach involved splitting the training set into two, good quality fields and bad quality fields.

Good quality data included quality 2 and 3
bad quality data included quality 1 fields.
Only good quality data was used for validation as the test set had only quality 2 and 3 fields.

Vegetation features used in this approach included : WDRVI, GNDVI, NDVI and NDRE only. Raw image pixel data was not used in training. Lightgbm was trained across five-folds with a CV score of 1.66 and a private LB score of 1.64