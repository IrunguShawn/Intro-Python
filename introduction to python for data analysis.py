# %%
#IMPORTING BANK DATASET
import pandas as pd
bank = pd.read_csv(r'C:\Users\NJERI SHAWN\Desktop\python pdf\Bank.csv')
print(bank)
bank['Salary'].mean() #same as sal.mean() after assigning sal=bank['salary']
sal=bank['Salary']
sal.mean()
sal.min()
sal.max()
sal.median()
sal.std()
#or we can get all statistical summary of all numeric data in the dataset by using the describe() command
bank.describe()


# %%
sal.mean(),sal.min(),sal.max(), sal.std()

# %%
sal.describe()

# %%
bank['Salary'].describe()

# %%
import seaborn as sns
sns.histplot(x=bank['Salary'])

# %%
sns.histplot(x=bank['Salary'], bins=10, kde=True);

# %%
sns.histplot(x=bank['Salary'],bins=10, kde=True,color='brown');

# %% [markdown]
# 

# %%
sns.boxplot(y=bank['Salary']);

# %%
sns.boxplot(y=bank['Salary'],color='lightgreen',showmeans=True);

# %%
#Filtering Data in Python

# %%
bank['Gender']=='Female'

# %%
bank[bank['Gender']=='Female']

# %%
FemaleEmployees=bank[bank['Gender']=="Female"]
type(FemaleEmployees)

# %%
round(FemaleEmployees['Salary'].mean(),2)

# %%
(bank['Gender']=='Female')&(bank['JobGrade']==1)

# %%
bank[(bank['Gender']=='Female')&(bank['JobGrade']==1)].shape

# %%
bank[bank['JobGrade']>=4]

# %%
mgmt=[4,5,6]
bank[bank['JobGrade'].isin(mgmt)]

#isin() method checks if the dataframe contains the specified values

# %%
#Recoding Data in Python

# %%
#Adding a Column in our Dataset
bank['Dummy']=0
bank.head()

# %%
#Dropping a Column in our Dataset
bank.drop('Dummy',axis=1, inplace=True)
bank.head()

# %%
#Recoding using the numpy where method
import numpy as np
bank['GenderDummy_F'] = np.where(bank['Gender']=="Female",1,0)
bank.head()

# %%
#Recoding Using the apply() Function
#The easiest way to see how this works is to start with a parameterized function that implements theif/then logic. What follows is a standard function declaration in Python. The code defi nes a new function called“my_recode” which takes a single parameter “gender”. The function returns a 1 or 0 depending on the value passed toit:

def my_recode(gender):
    if gender == "Female":
        return 1
    else:
        return 0

# %%
my_recode("Female"), my_recode("Male")

# %%
bank['GenderDummy_F']=bank['Gender'].apply(my_recode)
bank.head()

# %%
#Recoding Using a Lambda Function
bank['GenderDummy_F']=bank['Gender'].apply(lambda x: 1 if x == "Female" else 0)
bank.head()

# %%
#Replacing Values from a List
grades=[1,2,3,4,5,6]
status=["non-mgmt","non-mgmt","non-mgmt","non-mgmt","mgmt","mgmt"]
bank['Manager']=bank['JobGrade'].replace(grades,status)
bank.head()

# %%
genders=["Female", "Male"]
dummy_vars=[1,0]
bank['GenderDummy_F'] = bank['Gender'].replace(genders, dummy_vars)
bank.head()

# %%
#Logging variables
bank['logSalary']=np.log(bank['Salary'])
bank.head()

# %%
import seaborn as sns
sns.kdeplot(x=bank['logSalary'],fill=True,linewidth=2);

# %% [markdown]
# ## Gap analysis with Continuous Variables

# %%

# Recall the purpose of Gap Analysis: determine whether two samples of data are different. In our running example, we want to determine whether Sample 1 (salaries of female employees in the bank) is different from Sample 2 (salaries of male employees at the bank). We generally come at Gap Analysis in two steps:
        # 1. Plot the data in such a way that we can visually assess whether a gap exists. These visualizations also come in handy later when communicating the results of any formal analysis.
        # 2. Conduct a formal gap analysis using statistical techniques

#Using BoxPlots
#ensure Seaborn is loaded
import seaborn as sns
sns.boxplot(x=bank['Salary'], y=bank['Gender'], showmeans=True);

# %%
#As an aside: We can do the same kind of analysis by “JobGrade”. But recall that we left JobGrade as an integer and did not convert it to a category variable (as we did for Gender and PCJob). We can make this conversion on the fly in order to get a boxplot:
sns.boxplot(x=bank['Salary'], y=bank['JobGrade'].astype('category'), showmeans=True);



# %%
# Faceted histograms
# We used the notion of a “facet” in R to create a grid of histograms. In this case, we want a grid with one column and tworows. The rows correspond to different values of the “Gender” variable. This is a bit easier in Python with Seaborn’s displot function, which creates a faceted distribution plot. Here the row argument tells Seaborn to create one row foreach value of gender. I have also set the linewidth property to zero and added kernel density plots

sns.displot (x='Salary',row='Gender',data=bank,linewidth=0,kde=True);

# %%
#  As mentioned in our discussion of R, overlaying histograms almost never makes sense—the result is typically a mess,which is why SAS Enterprise Guide stacks them one on top of the other (as we just did above). Here is an overlayedhistogram for the bank salary data. Note that the color are added, so we get a third color in regions of overlap:

sns.histplot(x='Salary',hue='Gender',data=bank,linewidth=0);

# %%
# A better approach is to stack kernel density plots. Note that I have also added some shading to make the result lookmarginally cooler

sns.kdeplot(x='Salary',hue='Gender',data=bank,fill=True);

# %% [markdown]
# ## T-tests

# %%
# We use the t-test at this point to formally test the hypothesis that two distributions have the same sample mean (and thus are “the same”—or at least close enough). As in Excel and R, the two main preconditions to running the test inPython are:
    # 1.Getting the data in the right format
    # 2.Determining which version of the t-test to run: equal variance or unequal variance

female_sal=bank[bank['Gender']=="Female"]['Salary']
male_sal=bank[bank['Gender']=="Male"]['Salary']
female_sal

# Testing for equality of varianceUsing Levene's Test
# ensure the scipy stats module is loaded

from scipy import stats
stats.levene(female_sal,male_sal)


# from the  levene results below , we seen that the version of t-test to run is of unequal variance since p value has e-07, meaning it is smaller enough to treat as zero

# %% [markdown]
# # Running the t-test to compare the means

# %%

import statsmodels.stats.api as sms
model=sms.CompareMeans.from_data(bank[bank['Gender']=="Female"]['Salary'],bank[bank['Gender']=="Male"]['Salary'])
model.summary(usevar='unequal')


# %%
import statsmodels.stats.api as sms
model=sms.CompareMeans.from_data(female_sal,male_sal)
model.summary(usevar='unequal')



# %%



