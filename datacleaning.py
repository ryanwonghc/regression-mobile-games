#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:18:06 2020

@author: Ryan Wong
"""

# Import all necessary libraries
import pandas as pd
import re
from datetime import datetime, date

# Read in data
df = pd.read_csv('appstore_games.csv')

# Remove unneccesary columns: URL, ID, Icon URL.
df = df.drop(columns='URL')
df = df.drop(columns='ID')
df = df.drop(columns='Icon URL')

#Remove entries without any rating data
df = df[df['Average User Rating'].notna()]

# Remove entries with less than 100 ratings for credible reviews
df = df[df['User Rating Count'] >= 100]

# Describe data to better understand how to categorize data
desc = df. describe()

# Includes subtitle? (y/n)
df['Subtitle'] = df['Subtitle'].fillna('') #No data will be taken as empty string
df['subtitle_yes_no'] = df['Subtitle'].apply(lambda x: 0 if x == '' else 1)

# In app purchases categories and dummy variables
df['In-app Purchases'] = df['In-app Purchases'].fillna('0.0') #No data will be taken as $0
df['In-app Purchases'] = df['In-app Purchases'].apply(lambda x: '0.0' if x == '0' else x) #for consistency
df['In-app Purchases'] = df['In-app Purchases'].apply(lambda x: x.split(', ')) #turn string into list

# Prices range from 0 to 99.99. Split into 4 quadrants
df['In-App-Q1'] = df['In-app Purchases'].apply(lambda x: 1 if any(float(i) < 25 for i in x) else 0) #prices from 0 to 24.99
df['In-App-Q2'] = df['In-app Purchases'].apply(lambda x: 1 if any(25 <= float(i) < 50 for i in x) else 0) #prices from 25 to 49.99
df['In-App-Q3'] = df['In-app Purchases'].apply(lambda x: 1 if any(50 <= float(i) < 75 for i in x) else 0) #prices from 50 to 74.99
df['In-App-Q4'] = df['In-app Purchases'].apply(lambda x: 1 if any(75 <= float(i) < 100 for i in x) else 0) #prices from 75 to 99.99

# Num of words in description
df['Num_words_description'] = df['Description'].apply(lambda x: len(re.findall('(\w+)',str(x))))

# Age Rating. There are 4 categories: 4+, 9+, 12+, 17+.
# For dummy variables, we have to consider that 9+, 12+, and 17+ games are also 4+ games etc.
# We should make categories < 9, < 12, and < 17; 4+ games satisfy < 9, 4+ and 9+ games satisfy < 12, etc.
df['< 9'] = df['Age Rating'].apply(lambda x: 1 if x == '4+' else 0)
df['< 12'] = df['Age Rating'].apply(lambda x: 1 if x == '4+' or '9+' else 0)
df['< 17'] = df['Age Rating'].apply(lambda x: 1 if x == '4+' or '9+' or '12+' else 0)

# Num of languages
df['Languages'] = df['Languages'].fillna('') #No data will be taken as empty string
# Assume no data is one language offered
df['num_lang'] = df['Languages'].apply(lambda x: len(x.split(', ')))

# Size of game
# 25% of game size is 4.65e07, 50% is 1.12e08,75% is 2.21e08. Split into 4 categories
df['size_Q1'] = df['Size'].apply(lambda x: 1 if x < 4.65e7 else 0)
df['size_Q2'] = df['Size'].apply(lambda x: 1 if 4.65e7 <= x < 1.12e8 else 0)
df['size_Q3'] = df['Size'].apply(lambda x: 1 if 1.12e8 <= x < 2.21e8 else 0)
df['size_Q4'] = df['Size'].apply(lambda x: 1 if x >= 2.21e8 else 0)

# Genre dummy variables
genre = df['Genres'].str.get_dummies(sep=', ') #dummy variables for each genre category
genre = genre.drop(columns='Games') #Remove 'Games' category because that is the genre of every entry
df = df.merge(genre, left_index = True, right_index = True)

# Time since last update (in days, as of April 20, 2019)
df['Current Version Release Date'] = df['Current Version Release Date'].apply(lambda x: '0'+x if len(x)<10 else x) #make date 0 padded
df['Current Version Release Date'] = df['Current Version Release Date'].apply(lambda x: x[:6] + x[8:]) #remove century from date
df['update_date_object'] = df['Current Version Release Date'].apply(lambda x:  datetime.strptime(x, '%d/%m/%y').date())
df['days_since_update'] = df['update_date_object'].apply(lambda x: str(date(2020,4,20)-x)) #calculate difference in days
df = df.drop(columns='update_date_object') #delete column of date objects
df['days_since_update'] = df['days_since_update'].apply(lambda x: int(x.split(' ')[0])) #change days to int


# Remove all categories that have been transformed into dummy/categorical variables and any others that will not be used in model
# Name, Subtitle, User Rating Count, In-app Purchases, Description, Developer, Age Rating, Languages
# Size, Primary Genre, Genres, Original Release Date, Current Version Release Date

df = df.drop(columns='Name')
df = df.drop(columns='Subtitle')
df = df.drop(columns='User Rating Count')
df = df.drop(columns='In-app Purchases')
df = df.drop(columns='Description')
df = df.drop(columns='Developer')
df = df.drop(columns='Age Rating')
df = df.drop(columns='Languages')
df = df.drop(columns='Size')
df = df.drop(columns='Primary Genre')
df = df.drop(columns='Genres')
df = df.drop(columns='Original Release Date')
df = df.drop(columns='Current Version Release Date')

# Export Data Frame to csv
df.to_csv('appstore_games_cleaned.csv', index=False)
