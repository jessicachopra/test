#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:57:59 2017

@author: Jessica
"""

from __future__ import division, print_function
from future_builtins import ascii, filter, hex, map, oct, zip

import nltk  # draw on the Python natural language toolkit
from nltk.corpus import PlaintextCorpusReader

import os
import re
import pandas as pd
import numpy as np
import csv
import nltk  # draw on the Python natural language toolkit
import matplotlib.pyplot as plt  # 2D plotting
import statsmodels.api as sm  # logistic regression
import statsmodels.formula.api as smf  # R-like model specification
import patsy  # translate model specification into design matrices
from sklearn import svm  # support vector machines
from sklearn.ensemble import RandomForestClassifier  # random forests
import python_utilities
import datetime
from datetime import datetime
# import user-defined module
from python_utilities import evaluate_classifier, get_text_measures,\
    get_summative_scores

##############################################################################
##############################################################################
############################################################################
# read in positive and negative word lists from Hu and Liu (2004)

my_directory = '/Users/Jessica/Desktop/452/'    
positive_list = PlaintextCorpusReader(my_directory, 'Hu_Liu_positive_word_list.txt')
negative_list = PlaintextCorpusReader(my_directory, 'Hu_Liu_negative_word_list.txt', encoding='ISO-8859-1')
positive_words = positive_list.words()
negative_words = negative_list.words()





################################################################################    

# define bag-of-words dictionaries 
def bag_of_words(words, value):
    return dict([(word, value) for word in words])
positive_scoring = bag_of_words(positive_words, 1)
negative_scoring = bag_of_words(negative_words, -1)

scoring_dictionary = dict(positive_scoring.items() + negative_scoring.items())    

##############################################################################


###############################################################################
codelist = ['\r', '\n', '\t','\]','\[']  

# there are certain words we will ignore in subsequent
# text processing... these are called stop-words 
# and they consist of prepositions, pronouns, and 
# conjunctions, interrogatives, ...
# we begin with the list from the natural language toolkit
# examine this initial list of stopwords

nltk.download('stopwords')


# let's look at that list 
#print(nltk.corpus.stopwords.words('english'))


# previous analysis of a list of top terms showed a number of words, along 
# with contractions and other word strings to drop from further analysis, we add
# these to the usual English stopwords to be dropped from a document collection
more_stop_words = ['cant','didnt','doesnt','dont','goes','isnt','hes',\
    'shes','thats','theres','theyre','wont','youll','youre','youve', 'br'\
    've', 're', 'vs'] 

some_proper_nouns_to_remove = ['rating', 'date', 'twitter', 'https','retweeted'
                               ,'hashtag','retweet']


# start with the initial list and add to it 
stoplist = nltk.corpus.stopwords.words('english') + more_stop_words +\
    some_proper_nouns_to_remove

###############################################################################
#Parse text of the tweets and remove chracters or words not used for sentiment analysis

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



#############################################################################
###############################################################################
##  Load TWITTER data
###############################################################################
#read twitter data and create a dataframe with the date and tweets columns
def load_twitter_data(file):
    twtr_data_path = "/Users/Jessica/Desktop/452/"
    twtr_sonic_df = pd.read_csv(twtr_data_path+ file, delimiter = ',',index_col=None, header=None, parse_dates=True)
    
    dfList1 = twtr_sonic_df[3].tolist()
    dfList2 = twtr_sonic_df[5].tolist()
    dt_lst=[]
    m_lst = []
    for i in dfList1: 
       
        dt = datetime.strptime(i, '%m/%d/%y')
        m_lst.append(dt.month)
        dt=datetime.strftime(dt,'%m/%d/%y')
       
        dt_lst.append(dt)
        
    
    sonic = pd.DataFrame(
    {'Date': dt_lst,
     'comments': dfList2,
     'month':m_lst
     
    })
  
#aggregate tweets by month    
    aggText =  pd.DataFrame(sonic.groupby(['month'])['comments'].apply(list)) 
    return aggText
   
##################################################################################
##  Sentiment Scoring -
###############################################################################
def compute_sentiment_score(blogstring, scoring_dictionary):    
    # Because our interest is sentiment about Sonic,
    blogcorpus = blogstring.split()
    #print(blogcorpus)
    search_word = 'burger'
   
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
    
    srch_str_relevant_word_count = 15
    
    for iword in range(len(blogcorpus)):
        if blogcorpus[iword] == search_word:
            #print(blogcorpus[iword])
            groupcount = groupcount + 1
            #print(groupcount)
            for index in range(max(0,(iword - srch_str_relevant_word_count)),min((iword + srch_str_relevant_word_count+1), len(blogcorpus))):
                blogrelevant[index] = 1
                blogrelevantgroup[index] = groupcount
    
    # Compute the average sentiment score for the words nearby the search term.
    #print(blogrelevant)
    #print(blogscore)
    print('Average Sentiment Score Around Search Term:', search_word)
    search_term_SSA = round(sum((np.array(blogrelevant) * np.array(blogscore))) / sum(np.array(blogrelevant)),3)
    #print(round(sum((array(blogrelevant) * array(blogscore))) / sum(array(blogrelevant)),3))

    
    #print(search_term_SSA)
    cnt = blogstring.count(search_word)  
    #print(cnt)


    return(search_term_SSA, corpus_SSA)

##############################################################################
def scores(aggText):
    parseText = []  
    scoreMonth = []
    corpus_score = []
    search_term_score = []
    
    for index, row in aggText.iterrows():
           comments = text_parse(row['comments'])
           print(index)
           month = index
           #blogcorpus = comments.split()
           searchScore, corpusScore = compute_sentiment_score(comments, scoring_dictionary)
           parseText.append(comments )
           scoreMonth.append(month)
           corpus_score.append(corpusScore)
           search_term_score.append(searchScore)
    
    masterDf = pd.DataFrame({'comments': parseText, 'month' : scoreMonth, 'score': corpus_score, 'search_term': search_term_score})
    #print(masterDf)
    return masterDf
    


##############################################################################

def plot_scores(df1,df2, store):
            plt.plot(df1['month'], df1['score'], label='BurgerKing Corpus ', color = "blue")
            plt.plot(df2['month'], df2['score'], label='Sonic Corpus', color = "red")
            plt.plot(df1['month'], df1['search_term'], label='BurgerKing search_term ', color = "yellow")
            plt.plot(df2['month'], df2['search_term'], label='Sonic Search Term', color = "purple")
            plt.ylabel('Score')
            plt.xlabel('Monthly Scores')
            plt.title("Sonic Sentiment Score")
            plt.legend(loc='upper center', shadow=True)
            plt.grid(True)
            plt.xticks(rotation = 30)
            
            plt.show()
            #plt.figure()
            plt.savefig( 'sonic_sentiment.pdf')
            #plt.savefig(sentiment_out_path+ 'amzon_google_sentiment.pdf')
            plt.close()
            #plt.savefig('sample_plot.pdf', bbox_inches = 'tight', dpi = None,
                        #facecolor = 'w', edgecolor = 'b', orientation = 'portrait', 
                        #papertype = None, format = None, pad_inches = 0.25, frameon = None)


 
##############################################################################
def main():


    aggText = load_twitter_data("twitter_BurgerKing_hashtag_twitter.com.csv")
    masterdf1 = scores(aggText)
    aggText = load_twitter_data("twitter_hashtag_twitter.com.csv")
    masterdf2 = scores(aggText)
    plot_scores(masterdf1, masterdf2, "Sonic")
   
###############################################################################
   



##############################################################################   
if __name__ == '__main__':
    main()




