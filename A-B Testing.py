import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#######################################
#data reading and preprocessing
#######################################
df_control = pd.read_excel("datasets/ab_testing.xlsx", sheet_name = "Control Group")

df_test = pd.read_excel("datasets/ab_testing.xlsx", sheet_name = "Test Group")

df_control.head()
#i want to get rid of columns that are called "unnamed".
#so i'll redefine the dataframe with 4 variables.
df_control = df_control[["Impression", "Click", "Purchase", "Earning"]]
df_control.head()

#same for the test set.
df_test = df_test[["Impression", "Click","Purchase", "Earning"]]
df_test.head()



# our average earnings by clicks:

df_control.groupby("Click").agg({"Earning" : "mean"}).sort_values("Earning", ascending = False).head()
"""
Click      Earning         
4667.20523 2497.29522
6653.84552 2456.30424
6090.07732 2311.27714
3747.15754 2256.97559
4736.35337 2254.56383
"""
df_test.groupby("Click").agg({"Earning" : "mean"}).sort_values("Earning", ascending = False).head()

"""
            
Click       Earning        
4468.26679 3171.48971
4800.06832 2967.51839
3635.08242 2929.40582
4119.21862 2832.58467
3622.93635 2811.50273

"""


#the results showed that the new system provided a more successful outcome.
#but is it statistically significant? is there a real difference?


########################################
#Defining the Hypothesis
########################################

# H0 : M1=M2 (There is no statistically significant difference between the maximum bidding and the average bidding.)
# H1 : M1!=M2 (..there is.)

#############################################
#Assumption of Normality
############################################

# H0: Normal distribution assumption is provided.
# H1:..not provided.

# we reject H0 from 0.05 if p-value <.
# 0.05 H0 CANNOT BE REJECTED unless p-value !< 0.05.

test_stat, pvalue = shapiro(df_control["Click"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat,pvalue))
test_stat, pvalue = shapiro(df_test["Click"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9844, p-value = 0.8461
#Test Stat = 0.9896, p-value = 0.9699
#we can't reject H0. now we'll look at the assumption of homogeneity of variance.


#########################################
# Assumption of Homogeneity of Variance
#########################################

#H0: Assumption of Homogeneity of Variance is provided.
#H1: ...not provided.

# we reject H0 from 0.05 if p-value <.
# 0.05 H0 CANNOT BE REJECTED unless p-value !< 0.05.
#we will use levene test.

test_stat, pvalue = levene(df_control["Click"],
                           df_test["Click"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test Stat = 6.3041, p-value = 0.0141

#H0 REJECTED.
#So Assumption of Homogeneity of Variance is not provided.

#######################################################
#Application of hypothesis with t test
#######################################################

test_stat, pvalue = ttest_ind(df_control["Click"],
                              df_test["Click"],
                              equal_var = False)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test Stat = 4.4266, p-value = 0.0000
#H0 rejected, so it can't be said that there is no significant difference between the average bidding and the maximum bidding.
#new feature seems better.
#We used T-test, Shapiro and Levene tests. We used a parametric T test, since the distribution of normality was provided.

#############################
#Purchase
#############################

df_control["Purchase"].mean()
#550.89
df_test["Purchase"].mean()
#582.10

#we have a %0.04 more purchase with the new changes. but is it statistically significant?

##########################
#defining the hypothesis
##############################
# H0 : M1=M2 (There is no statistically significant difference between the maximum bidding and the average bidding.)
# H1 : M1!=M2 (..there is.)


###################
#Assumption of Normality
##################

# H0: Normal distribution assumption is provided.
# H1:..not provided.

# we reject H0 from 0.05 if p-value <.
# 0.05 H0 CANNOT BE REJECTED unless p-value !< 0.05.

test_stat, pvalue = shapiro(df_control["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test Stat = 0.9773, p-value = 0.5891
#Test Stat = 0.9589, p-value = 0.1541
#we can't reject H0. now we'll look assumption of homogeneity of variance.

###############################
#Assumption of homogeneity of variance
###############################

#H0: Assumption of Homogeneity of Variance is provided.
#H1: ...not provided.

# we reject H0 from 0.05 if p-value <.
# 0.05 H0 CANNOT BE REJECTED unless p-value !< 0.05.


test_stat, pvalue = levene(df_control["Purchase"],
                           df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test Stat = 2.6393, p-value = 0.1083
#Assumption of Homogeneity of Variance is provided.


#######################################################
#Application of hypothesis with t test
#######################################################
test_stat, pvalue = ttest_ind(df_control["Purchase"],
                              df_test["Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test Stat = -0.9416, p-value = 0.3493
#We can't reject H0.
#We did not reject H0, we can say that there is no statistically significant (95% significant) difference.
#Since we could not achieve statistical significance in "purchases", it would be best to continue to observe the process and collect more data.
