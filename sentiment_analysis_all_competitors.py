# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:33:38 2017

@author: Oyumaa
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 2D plotting
import datetime
from datetime import datetime
# import user-defined module
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
import nltk


# directory
directory='C:/MSPA/Capstone/Project/Text Mining From Social Media/'

###############################################################################


# load twitter data
def load_twitter_data(twitter_file):
    twtr_df = pd.read_csv(os.path.join(directory,twitter_file), delimiter = ',',index_col=None, header=None, parse_dates=True)
    
    dfList1 = twtr_df[3].tolist()
    dfList2 =twtr_df[5].tolist()
    dt_lst=[]
    for i in dfList1: 
        i=i.strip(".")
        dt = datetime.strptime(i, '%m/%d/%y')
        dt=datetime.strftime(dt,'%m/%d/%y')
        dt_lst.append(dt)
   
    
    twitter_df = pd.DataFrame(
    {'Date': dt_lst,
     'comments': dfList2,
     
    })

    twitter_df['Date']=pd.to_datetime(twitter_df['Date'])
    twitter_df['YearMonth'] =twitter_df['Date'].apply(lambda x:x.strftime('%Y%m'))
    
    return twitter_df


###############################################################################


# load facebook data

def load_facebook_data(facebook_file):

    facebook_df = pd.read_csv(os.path.join(directory,facebook_file), encoding='latin-1')
    facebook_df['status_published']=pd.to_datetime(facebook_df['status_published'])
    facebook_df['YearMonth'] =facebook_df['status_published'].apply(lambda x:x.strftime('%Y%m'))

    return facebook_df


    
    

###############################################################################
## Consolidate twitter and facebook date
###############################################################################

def consolidate_text_data(twitter_file, facebook_file): 
    df1=load_facebook_data(facebook_file)
    df2=load_twitter_data(twitter_file)
    df1=df1.rename(columns={'status_message':'comments'})
    df1=df1[['YearMonth','comments']]
    df2=df2[['YearMonth','comments']]
    
    consolidated=df1.append(df2)
    comments_by_date=pd.DataFrame(consolidated.groupby(['YearMonth'])['comments'].apply(list)) 

    return comments_by_date


    

###############################################################################
## facebook date
###############################################################################

def get_text_data(facebook_file): 
    df1=load_facebook_data(facebook_file)
    df1=df1.rename(columns={'status_message':'comments'})
    df1=df1[['YearMonth','comments']]
   
    comments_by_date=pd.DataFrame(df1.groupby(['YearMonth'])['comments'].apply(list)) 

    return comments_by_date
    
    
    
    


###############################################################################
##  Load scoring dictionary
###############################################################################

# define bag-of-words dictionaries 
my_directory = 'C:/MSPA/452_Web Analytcs/Assignment/Individual Assignment 4/Sentiment Analysis/000_sentiment_jump_start/'
positive_list = PlaintextCorpusReader(my_directory, 'Hu_Liu_positive_word_list.txt')
negative_list = PlaintextCorpusReader(my_directory, 'Hu_Liu_negative_word_list.txt',encoding='latin-1')

positive_words = positive_list.words()
negative_words = negative_list.words()

# define bag-of-words dictionaries 
def bag_of_words(words, value):
    return dict([(word, value) for word in words])
    
    
positive_scoring = bag_of_words(positive_words, 1)
negative_scoring = bag_of_words(negative_words, -1)
scoring_dictionary = dict(positive_scoring.items()| negative_scoring.items())


###############################################################################

codelist = ['\r', '\n', '\t','\]','\[']  

# there are certain words we will ignore in subsequent
# text processing... these are called stop-words 
# and they consist of prepositions, pronouns, and 
# conjunctions, interrogatives, ...
# we begin with the list from the natural language toolkit
# examine this initial list of stopwords

nltk.download('stopwords')

# previous analysis of a list of top terms showed a number of words, along 
# with contractions and other word strings to drop from further analysis, we add
# these to the usual English stopwords to be dropped from a document collection
more_stop_words = ['cant','didnt','doesnt','dont','goes','isnt','hes',\
    'shes','thats','theres','theyre','wont','youll','youre','youve', 'br'\
    've', 're', 'vs'] 

some_proper_nouns_to_remove = ['rating', 'date', 'twitter', 'https','retweeted'
                               ,'hashtag','retweet']


###############################################################################
## start with the initial list and add to it for movie text work 

stoplist = nltk.corpus.stopwords.words('english') + more_stop_words +\
    some_proper_nouns_to_remove


    
    
###############################################################################
##  cleanse texts
###############################################################################
    


def text_parse(string):
    # replace non-alphanumeric with space
    string_of_lists = ' '
    for i in string:
     string_of_lists += str(i)
    temp_string = re.sub('[^a-zA-Z]', '  ', string_of_lists)    
    # replace codes with space
    for i in range(len(codelist)):
        stopstring = ' ' + codelist[i] + '  '
        temp_string = re.sub(stopstring, '  ', temp_string)      
    # replace single-character words with space
    temp_string = re.sub('\s.\s', ' ', temp_string)   
    # convert uppercase to lowercase
    temp_string = temp_string.lower()    
    # replace selected character strings/stop-words with space
    for i in range(len(stoplist)):
        stopstring = ' ' + str(stoplist[i]) + ' '
        temp_string = re.sub(stopstring, ' ', temp_string)        
    # replace multiple blank characters with one blank character
    temp_string = re.sub('\s+', ' ', temp_string)    
    return(temp_string)   
   
##################################################################################
##  Sentiment Scoring - SEARCH word 
###############################################################################
def compute_sentiment_score(blogstring, scoring_dictionary):    
    # Because our interest is sentiment about Sonic,
    blogcorpus = blogstring.split()
    search_word = 'food'
   
    # list for assigning a score to every word in the blogcorpus
    # scores are -1 if in negative word list, +1 if in positive word list
    # and zero otherwise. We use a dictionary for scoring.
    blogscore = [0] * len(blogcorpus)  # initialize scoring list
    
    for iword in range(len(blogcorpus)):
        if blogcorpus[iword] in scoring_dictionary:
            blogscore[iword] = scoring_dictionary[blogcorpus[iword]]
            
    # report the norm sentiment score for the words in the corpus
    print('Corpus Average Sentiment Score:')
    #corpus_SSA = round(sum(blogscore) / (len(blogcorpus) - blogstring.count('xxxxxxxx')), 3)
    corpus_SSA = round(sum(blogscore) / (len(blogcorpus)), 3)
    #print(corpus_SSA)        
    
    # Read the blogcorpus from beginning to end
    # identifying all the places where the search_word occurs.
    # We arbitrarily identify search-string-relevant words
    # to be those within three words of the search string.
    blogrelevant = [0] * len(blogcorpus)  # initialize blog-relevnat indicator
    blogrelevantgroup = [0] * len(blogcorpus)
    groupcount = 0  
    
    srch_str_relevant_word_count = 2
    
    for iword in range(len(blogcorpus)):
        if blogcorpus[iword] == search_word:
            groupcount = groupcount + 1
            for index in range(max(0,(iword - srch_str_relevant_word_count)),min((iword + srch_str_relevant_word_count+1), len(blogcorpus))):
                blogrelevant[index] = 1
                blogrelevantgroup[index] = groupcount
    
    # Compute the average sentiment score for the words nearby the search term.
    print('Average Sentiment Score Around Search Term:', search_word)
    search_term_SSA = round(sum((np.array(blogrelevant) * np.array(blogscore))) / sum(np.array(blogrelevant)),3)
    print(search_term_SSA)
    cnt = blogstring.count(search_word)  
    print(cnt)


    return(corpus_SSA)


###############################################################################
##  score sentiment by month
###############################################################################

def scores(df):
    parseText = []  
    scoreMonth = []
    score_cnt = []
    
    for index, row in df.iterrows():
           comments = text_parse(row['comments'])
           print(index)
           month = index
           blogcorpus = comments.split()
           blogscore = compute_sentiment_score(comments, scoring_dictionary)
           parseText.append(comments)
           scoreMonth.append(month)
           score_cnt.append(blogscore)
    
    masterDf = pd.DataFrame({'comments': parseText, 'date' : scoreMonth, 'score': score_cnt})
    return masterDf



    
##############################################################################    
## Get data for monthly sentiment score for all companies      
    
    
sonic_data = consolidate_text_data('twitter_Sonic_Data.csv','SonicDriveIn_facebook_statuses.csv')
burgerking_data = consolidate_text_data('twitter_BurgerKing_Data.csv','burgerking_facebook_statuses.csv')
wendys_data = consolidate_text_data('twitter_Wendys_Data.csv','Wendys_facebook_statuses.csv')
jackinthebox_data = consolidate_text_data('twitter_Jackinthebox_Data.csv','jackinthebox_facebook_statuses.csv')
macdonalds_data = consolidate_text_data('twitter_McDonalds_Data.csv','MacDonalds_facebook_statuses.csv')





sonic_score = scores(sonic_data)
burgerking_score = scores(burgerking_data)
wendys_score = scores(wendys_data)
jackinthebox_score = scores(jackinthebox_data)
macdonalds_score = scores(macdonalds_data)


###############################################################################

## MacDonalds has data from 2014-07 so  we need to review data from 2014-04.
###############################################################################
# convert data type for the date and time
burgerking_score['date']=pd.to_datetime(burgerking_score['date'], format='%Y%m')
sonic_score['date']=pd.to_datetime(sonic_score['date'], format='%Y%m')
wendys_score['date']=pd.to_datetime(wendys_score['date'], format='%Y%m')
jackinthebox_score['date']=pd.to_datetime(jackinthebox_score['date'], format='%Y%m')
macdonalds_score['date']=pd.to_datetime(macdonalds_score['date'], format='%Y%m')



# take all data from 2014- 04
burgerking=burgerking_score[burgerking_score['date']>='2014-07-01']
sonic=sonic_score[sonic_score['date']>='2014-07-01']
wendys=wendys_score[wendys_score['date']>='2014-07-01']
jackinthebox=jackinthebox_score[jackinthebox_score['date']>='2014-07-01']
macdonalds=macdonalds_score[macdonalds_score['date']>='2014-07-01']


# plot the sentiment scores and compare

plt.plot(sonic['date'],sonic['score'], label='Sonic', color = "red")
plt.plot(burgerking['date'],burgerking['score'], label='BurgerKing', color = "blue")
plt.plot(wendys['date'],wendys['score'], label='Wendys', color = "yellow")
plt.plot(jackinthebox['date'],jackinthebox['score'], label='Jack in the Box', color = "green")
plt.plot(macdonalds['date'],macdonalds['score'], label='MacDonalds', color = "black")
plt.ylabel('Score')
plt.xlabel('Monthly Scores')
plt.title("Sentiment Score for search term FOOD")
plt.legend(loc='best', shadow=False, framealpha=0.0)
plt.xticks(rotation = 90)
plt.show()


# mean scores
mean_sonic=sonic['score'].mean()
mean_burgerking=burgerking['score'].mean()
mean_macdonalds=macdonalds['score'].mean()
mean_wendys=wendys['score'].mean()
mean_jackinthebox=jackinthebox['score'].mean()



