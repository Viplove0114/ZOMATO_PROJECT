# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 23:04:28 2022

@author: 91782
"""

import pandas as pd
import numpy as np

#reading the dataset
df = pd.read_csv('D:\\DATA_SCIENCE\\readmycourse\\assignments\\pandas_ass\\zomato_sorted.csv')
df.shape

#get a concise summary of the dataframe
df.info()

df['name'].nunique()
df['location'].unique()

#removing the duplicates 
df.duplicated().sum()
df.drop_duplicates(inplace=True)


#Remove the NaN values from the dataset
df.isnull().sum()
df.dropna(how='any',inplace=True)

#reading column names
df.columns

#changing column names
df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})

#Transformations
#getting those rows which don't have the value "NEW"
df = df.loc[df.rate != 'NEW']
#replacing '-'
df = df.loc[df.rate != '-'].reset_index(drop=True)
#replacing '/5'
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
df.rate = df.rate.apply(remove_slash).str.strip().astype(float)
# Discretizing the ratings into a categorical feature with 4 classes
df["rate"] = pd.cut(df["rate"], bins = [0, 3.0, 3.5, 4.0, 5.0], labels = ["0", "1", "2", "3"])


df['rate'].unique()
df['rate'].nunique()

#changing dtype to 'str'
df['cost'] = df['cost'].astype('str')
#replacing ','
df['cost']= df['cost'].apply(lambda x: x.replace(',','.'))
#changing dtype to 'float'
df['cost'] = df['cost'].astype('float')

#adjusting the column 'name'
df.name = df.name.apply(lambda x: x.title())
df.online_order.replace(('Yes','No'),(True,False),inplace=True)
df.book_table.replace(('Yes','No'),(True,False),inplace=True)

'''
################################################################################################################################################
#extracting the the text data for further text pre-processing
df_ob = df.iloc[:,[0,1,2,6,7,8,9,10,12,13,14,15]].copy()
df_ob.info()


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

ls = WordNetLemmatizer()

import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
df_ob["reviews_list"] = df_ob["reviews_list"].apply(lambda text: remove_punctuation(text))

## Removal of Stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
df_ob["reviews_list"] = df_ob["reviews_list"].apply(lambda text: remove_stopwords(text))


## Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
df_ob["reviews_list"] = df_ob["reviews_list"].apply(lambda text: remove_urls(text))


#NAMES OF ALL RESTAURANTS 
restaurant_names = list(df_ob['name'].unique())
restaurant_names

df.shape
df_ob.columns



from tqdm import tqdm
all_ratings=[]
for ratings in tqdm(df['reviews_list']):
    ratings = eval(ratings)
    
    for score, doc in ratings:
        if score:
            score = score.strip("Rated").strip()
            doc = doc.strip('RATED').strip()
            
            score = float(score)
            all_ratings.append([score, doc])

ratings_df = pd.DataFrame(all_ratings,columns=['score','doc'])
ratings_df.shape
ratings_df.head()

ratings_df['doc'] = ratings_df['doc'].apply(lambda text: remove_stopwords(text))

 ###########################################################################################################################################################################################
'''


#visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#numbers of restaurants in different locations
fig = plt.figure(figsize=(20,7))
loc = sns.countplot(x="location",data=df, palette = "Set1")
loc.set_xticklabels(loc.get_xticklabels(), rotation=90, ha="right")
plt.ylabel("Frequency",size=15)
plt.xlabel("Location",size=18)
loc
plt.title('NO. of restaurants in a Location',size = 20,pad=20)
plt.savefig("Restaurants in Location")

#frequency of types of restaurants
fig = plt.figure(figsize=(17,5))
rest = sns.countplot(x="rest_type",data=df, palette = "Set1")
rest.set_xticklabels(rest.get_xticklabels(), rotation=90, ha="right")
plt.ylabel("Frequency",size=15)
plt.xlabel("Restaurant type",size=15)
rest 
plt.title('Restaurant types',fontsize = 20 ,pad=20)
plt.savefig('Restaurant types')

#25 most famous restaurant chains in bengaluru
plt.figure(figsize=(15,7))
chains=df['name'].value_counts()[:25]
sns.barplot(x=chains,y=chains.index,palette='Set1')
plt.title("Most famous restaurant chains in Bangaluru",size=20,pad=20)
plt.xlabel("Number of outlets",size=15)
plt.savefig('Most famous restaurant chains')


#Restaurants delivering Online or not
sns.countplot(df['online_order'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or Not')
plt.savefig("online.png")

#Restaurants allowing table booking or not
sns.countplot(df['book_table'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.savefig("Book_Table.png")
plt.title('Restaurants allowing table booking or not')

#table booking vs rate
plt.rcParams['figure.figsize'] = (13, 9)
Y = pd.crosstab(df['rate'], df['book_table'])
Y.div(Y.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])
plt.title('table booking vs rate', fontweight = 30, fontsize = 20)
plt.legend(loc="upper right")
plt.savefig("Table_Booking_Rate.png")
plt.show()

#location
sns.countplot(df['city'])
sns.countplot(df['city']).set_xticklabels(sns.countplot(df['city']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(13,13)
plt.savefig("Location.png")
plt.title('Location')

#location and rating
loc_plt=pd.crosstab(df['rate'],df['city'])
loc_plt.plot(kind='bar',stacked=True);
plt.title('Location - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Location',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');
plt.legend().remove();
plt.savefig("Location Rating.png")

#restaurants types
sns.countplot(df['rest_type'])
sns.countplot(df['rest_type']).set_xticklabels(sns.countplot(df['rest_type']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.savefig("Restuarant Type")
plt.title('Restuarant Type')

#restaurants type vs rating
loc_plt=pd.crosstab(df['rate'],df['rest_type'])
loc_plt.plot(kind='bar',stacked=True);
plt.title('Rest type - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Rest type',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');
plt.legend().remove();
plt.savefig('Rest Type-Rating')

#types of service
sns.countplot(df['type'])
sns.countplot(df['type']).set_xticklabels(sns.countplot(df['type']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.title('Type of Service')
plt.savefig('Types of Service')

#rating and type
type_plt=pd.crosstab(df['rate'],df['type'])
type_plt.plot(kind='bar',stacked=True);
plt.title('Type - Rating',fontsize=15,fontweight='bold')
plt.xlabel('Type',fontsize=10,fontweight='bold')
plt.ylabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');
plt.savefig('Type and Rating')

#cost of restaurants
sns.countplot(df['cost'])
sns.countplot(df['cost']).set_xticklabels(sns.countplot(df['cost']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.title('Cost of Restuarant')
plt.savefig('Cost of Restaurant')


