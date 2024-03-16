#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:38:51 2024

@author: m1cmb07
"""

import os
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk import punkt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from os import listdir
from os.path import isfile, join
import string
import datetime
import pandas
from transformers import BertTokenizer, BertForSequenceClassification
import pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')



from transformers import BertTokenizer, BertForSequenceClassification

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


os.chdir('/msu/home/m1cmb07/Personal/MATH5152/project/')
 
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

speeches_dir = './fomc-hawkish-dovish-main/data/raw_data/speech/text/all/'

#os.chdir('/msu/home/m1cmb07/Personal/MATH5152/project/fomc-hawkish-dovish-main/data/raw_data/speech/text/all')


fomc_speeches = [f for f in listdir(speeches_dir) if isfile(join(speeches_dir, f))]


analyze = SentimentIntensityAnalyzer()



def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    #text = text.lower()
    #text = "".join([char.lower() for char in text if char not in string.punctuation])    

    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopWords]

    # Lemmatize the tokens
    #lemmatizer = WordNetLemmatizer()
    #lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    #processed_text = ' '.join(lemmatized_tokens)
    processed_text = ' '.join(tokens)

    return processed_text



speech_sentiments_pos = []
speech_sentiments_neg = []
speech_sentiments_neu = []
speech_sentiments_comp = []
for speech in fomc_speeches:
    f = open(speeches_dir + speech, 'r')
    speech_text = f.read()
    speech_text = preprocess_text(speech_text)
    sentiments = analyze.polarity_scores(speech_text)
    speech_sentiments_pos.append(sentiments['pos'])
    speech_sentiments_neg.append(sentiments['neg'])
    speech_sentiments_neu.append(sentiments['neu'])
    speech_sentiments_comp.append(sentiments['compound'])
    

dates = []
for filename in fomc_speeches:
    date = ''.join([char for char in filename if char.isdigit()])
    if len(date) == 9:
        date = date[0:len(date)-1]
    dates.append(date)

speech_dates = [datetime.datetime.strptime(date, '%Y%m%d') for date in dates]

meetings_df = pandas.read_csv('fomc-meeting-dates.csv')
meeting_df = meetings_df['press_release_date']

meeting_dates = [datetime.datetime.strptime(meeting, '%m/%d/%Y') for meeting in meeting_df if isinstance(meeting, str)]

