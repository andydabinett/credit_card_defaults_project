"""""
This project aims to predict how capable each client is of paying off their credit card balances. 
The dataset represents default and non-default accounts of credit card clients in Taiwan from 2005. 
Using this historical data, I will try to build a predictive model that classifies whether an account will pay 
off its next months balance or default. 

"""

# %% Cell 1
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns 

    
# %% Cell 2
#Read in CSV and do some basic EDA 
df = pd.read_csv('ccDefaults(1) copy.csv')
df.head()
df.describe()

# %% Cell 3
#Create correlation heatmap to show if there is correlation between variables 
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap')
plt.show()

#Data Preprocessing: 
print(df.isnull().sum())


# %% Cell 4
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

features = df.iloc[:, [1] + list(range(5, df.shape[1]-1))]
scaler.fit(features)
print(features)
# %% Cell 5
df_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(df_scaled, columns=features.columns)
print(df_scaled.head())

# %% Cell 6 
#Check for class imbalance 
df['dpnm'].value_counts()

# %% Cell 7
#Evaluate Balance and Age variables
plt.subplots(figsize=(20,5))
plt.subplot(121)
sns.distplot(df.LIMIT_BAL)

plt.subplot(122)
sns.distplot(df.AGE)

plt.show()

# %% Cell 8
plt.figure(figsize=(12, 6))
df[df['dpnm'] == 1]['AGE'].hist(alpha=0.5, color='blue', bins=30, label='Default=1')
df[df['dpnm'] == 0]['AGE'].hist(alpha=0.5, color='red', bins=30, label='Default=0')
plt.title('Age Distribution by Default Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.kdeplot(df[df['dpnm'] == 1]['AGE'], label='Default=1', shade=True)
sns.kdeplot(df[df['dpnm'] == 0]['AGE'], label='Default=0', shade=True)
plt.title('Age Distribution by Default Status')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

#Want to see the distribution of defaults with respect to age. 

#Observation: The distribution of defaults follows the same shape as the distribution of Ages. Meaning, age likely is not 
# a major predictor in defaulting. Otherwise, we would see some sort of skew. 


# %% Cell 8

#While we will not be using these attributes to prevent a discriminative model, 
#I still thought it might be interesting to see the distribution of defaults with respect to Sex, Education, and Marriage 


#SEX: 
df['SEX_label'] = df['SEX'].map({1: 'Male', 2: 'Female'})
plt.figure(figsize=(10, 6))
sns.countplot(x='SEX_label', hue='dpnm', data=df, palette='coolwarm')
plt.title('Default Status by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Default')
plt.show()

#EDUCATION: 
education_map = {
    1: 'Graduate School',
    2: 'University',
    3: 'High School',
    4: 'Others',
    5: 'Unknown',
    6: 'Unknown'
}

df['EDUCATION_label'] = df['EDUCATION'].map(education_map)

plt.figure(figsize=(12, 6))
sns.countplot(x='EDUCATION_label', hue='dpnm', data=df, palette='coolwarm', order=df['EDUCATION_label'].value_counts().index)
plt.title('Default Status by Education')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title='Default')
plt.xticks(rotation=45)
plt.show()

#MARRIAGE: 
marriage_map = {
    1: 'Married',
    2: 'Single',
    3: 'Other/Unknown',
}

df['MARRIAGE_label'] = df['MARRIAGE'].map(marriage_map)

plt.figure(figsize=(12, 6))
sns.countplot(x='MARRIAGE_label', hue='dpnm', data=df, palette='coolwarm', order=df['MARRIAGE_label'].value_counts().index)
plt.title('Default Status by Marriage')
plt.xlabel('Marriage Status')
plt.ylabel('Count')
plt.legend(title='Default')
plt.xticks(rotation=45)
plt.show()



# %% Cell 9
X = features #Going to use just Balance, Age, Pay, and Bill attributes, as these show higher correlation with the target variable. 
            #Additionally, we must be aware of the fact that building a model that predicts loan default based on Sex or Education 
            #could potentially be unfairly discriminatory towards certain sex's, marriage, or education levels. 
Y = df['dpnm']

feature_names = list(X.columns)
print(feature_names)


from sklearn.model_selection import train_test_split
x_temporary, x_test, y_temporary, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_temporary, y_temporary, test_size=0.2, random_state=42)


# %% Cell 10
#Handle the imbalanced dataset, which we noticed earlier
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)


