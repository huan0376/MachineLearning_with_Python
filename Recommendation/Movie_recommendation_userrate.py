# Source:  from Geeksforgeeks.
# Dataset: file (user_id, item_id, rating, timestamp) and movie list (item_id, title)
# Recommendation mechanism: content-based filtering. 
# Procedure: Build movie similarity matrix based on user rating for each movie.
#            if user U1 watches movie M1, system will recommend the Movies that has high similarity with movie M1.



import pandas as pd
import numpy as np



# Step 1: read two datasets.
# 1-1: read the user rating data from website.
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
path_userrate = 'https://cdncontribute.geeksforgeeks.org/wp-content/uploads/file.tsv'
df_userrate = pd.read_csv(path_userrate, sep = '\t', names = column_names)
print(df_userrate.head(5))


# 1-2: read the movie list data from website.
path_movielist = 'https://cdncontribute.geeksforgeeks.org/wp-content/uploads/Movie_Id_Titles.csv'
df_movielist = pd.read_csv(path_movielist)
print(df_movielist.head(5))



# Step 2: generate a new dataset using df_userrate and df_movielist.
data = pd.merge(df_userrate, df_movielist, on = 'item_id')
print(data.head())


# Step 3: generate a new dataset - rating specific movie including number of ratings and average of rating. 
# calculate the mean rating of all movies. 
print(data.groupby('title')['rating'].mean().sort_values(ascending=False).head())
print()
# calculate the number of rating of all movies. 
print(data.groupby('title')['rating'].count().sort_values(ascending=False).head())
print()

ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())
print(ratings.sort_values('num of ratings', ascending=False).head())


# Step 4: visualize the new dataset.
# 4-1: import the visualization library
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')

# 4-2: plot graph of 'number of ratings column'
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)

# 4-3: plot graph of average of rating - ratings
ratings['rating'].hist(bins=70)


# Step 5: create a new dataset - moviemat using pivot table.
moviepivot = data.pivot_table(index='user_id', columns='title', values='rating')
print(moviepivot.head(2))


# step 6: build similarity matrix among movie based on the ratings. 
# 6-1: analyze correlation with similar movie. take an example 'Star Wars (1977)'
starwars_user_ratings = moviepivot['Star Wars (1977)']
print(starwars_user_ratings.head(5))
print()

# 6-2: analyze the correlation of 'Star Wars (1977)' with other movies. 
similar_to_starwars = moviepivot.corrwith(starwars_user_ratings)
print(similar_to_starwars.sort_values(ascending=False).head())
print(type(similar_to_starwars))
print()
# 6-3: transform the similar_to_starwars and drop the NaN data. 
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head(10))



# Step 7: combine the corr_starwars data with the movies rating details (number of rating and mean of rating).
# build Recommend list for starwars based on the correlation with other movies and the number of rating. 
corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars.sort_values('Correlation', ascending=False).head(10))
print()
recomm_starwars = corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending=False)
print(recomm_starwars.head(10))


# step 8: build another recommend movie list for 'Liar Liar (1997)'
liarliar_user_lists = moviepivot['Liar Liar (1997)'] # select user rating for the movie
# calculate the correlation between the movie with other movie. 
similar_to_liarliar = moviepivot.corrwith(liarliar_user_lists) 
# transform to DataFrame type.
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation']) 
corr_liarliar.dropna(inplace=True) # drop the NA rows
# join the column 'num of ratings'
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
# generate the recommendating moviel list for liarliar
recomm_liarliar = corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending=False)
print(recomm_liarliar.head(5))

