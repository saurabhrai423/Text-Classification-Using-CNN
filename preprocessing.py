from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D , Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding , Concatenate
from tensorflow.keras.models import Model
import pickle
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import pandas


### Getting Labels
"""

import os
import pandas as pd

path = 'documents'
a = []
with os.scandir(path) as dirr:
    for file in dirr:
        a.append('_'.join(file.name.split('_')[:-1]))

data = pd.DataFrame()

data['class'] = a
print(data.head())
data.shape

"""### Preprocessing Email"""

import re
email = []
path = 'documents'
with os.scandir(path) as dirr:
    for file in dirr:
        y = ' '
        for i, line in enumerate(open(file)):
            
            match = (re.findall(r'[\w\.-]+@[\w\.-]+', line))
            if len(match) != 0:
                y += match[0] + ' '
        
        email.append(y)

new_email = []
for k in email:
    temp = ''
    mail = k.split(' ')
    for i in mail:
        if len(mail) != 0:
            temp += i[i.find('@')+1:] + ' '
    new_email.append(temp)
    
preprocessed_email = []
for k in new_email:
    k = k.replace('.' , ' ')
    k = k.replace('.com' , '')
    k = k.replace('com' , '')
    
    temp = ' '.join(word for word in k.split() if len(word)>2)
    
    preprocessed_email.append(temp)

len(preprocessed_email)

"""### Preprocessing Subject"""

sub = []

with os.scandir(path) as dirr:
    for file in dirr:
        for line in open(file):
            if 'Subject:' in line:
                sub.append(line[line.find('Subject:')+12:] )
                break

subject = []
for i in sub:
    subject.append(re.sub('\W+',' ', i) )

len(subject)

data['preprocessed_subject'] = subject

data.to_csv('data2.csv')

"""### Cleaning Text"""

# Removing first 3 lines from all text 
with os.scandir(path) as dirr:
    for file in dirr:
        lines = open(file).readlines()
        with open(file, 'w') as f:
            f.writelines(lines[3:])

import string
for i in range(len(email)):
    email[i] = email[i].strip()
    email[i] = email[i].split(' ')

# Getting Text to a list! 
path = 'documents'
text = []
with os.scandir(path) as dirr:
    for file in dirr:
        y = ' '
        for i, line in enumerate(open(file)):
            
                y += line + ' '
        
        text.append(y)

"""### Cleaning Text"""

def prep_text(text):
    
    pre_text = []
    mail = re.compile('[\w\.-]+@[\w\.-]+')   # Compiling email in regex
    tags = re.compile('<.*?>')                # Compiling tags text\\

    for line in text:

        line = line.strip()                     # Removing Spaces in start and end
        line = line.replace('  ' , '')        # Removing linegap in between text
        line = line.replace('\n' , '')          #Removing \n in text
        line = line.replace("\\", "")          #Removing \n in text
        line = line.replace('\t' , '')          #Removing \n in text
        line = line.replace('-' , '')          #Removing \n in text

        line = mail.sub('' , line)                          # Removing emails
        line = tags.sub('' , line)                         # Removing tags and text inbetween
        
        line = re.sub("[\(\[].*?[\)\]]", "", line)        # Removing Bracket text
        line = re.sub('\w+:\s?','',line)                  # Removing words ending with :

        line= re.sub(r"won't", "will not", line)
        line= re.sub(r"can\'t", "can not", line)
        line = re.sub(r"n\'t", " not",line)
        line = re.sub(r"\'re", " are",line)
        line= re.sub(r"\'s", " is", line)
        line= re.sub(r"\'d", " would", line)
        line = re.sub(r"\'ll", " will",line)
        line = re.sub(r"\'t", " not", line)
        line = re.sub(r"\'ve", " have", line)
        line= re.sub(r"\'m", " am", line)

        pre_text.append(line)

    return pre_text

preprocessed_text = prep_text(text)

len(preprocessed_text)

"""## Chunking """

import nltk
chunked = []

for line in preprocessed_text:
    
    word = nltk.word_tokenize(line)   
    pos_tag = nltk.pos_tag(word)   
    chunk = nltk.ne_chunk(pos_tag) 
    
    x  = ''
    for ne in list(chunk):
        if type(ne) is nltk.tree.Tree:
            if ne.label() not in ['PERSON']:
                x += ('_'.join([i[0] for i in ne.leaves()])) + ' '
        else:
            x += str(ne[0]) + ' '
            
    chunked.append(x)

"""### Last Preprocessing"""

start_ = re.compile('_+$')    # removes word starting with _
_end = re.compile('^_+')      # removes word end with _
digit = re.compile('\d+')
oneLetter_ = re.compile('^.{1}_.+')   
twoLetter_ = re.compile('^.{2}_.+')    #

prep_text = []
for line in list(chunked):
    
    line = start_.sub('', line)
    line = _end.sub('', line)
    line = digit.sub('', line)

    line = oneLetter_.sub(line[2:] , line)  # Removes first letter and underscore if '_' after 1 letter (exactly)
    line = twoLetter_.sub(line[3:], line)    #  Removes first ,second letter and underscore if '_' after 2 letters  (exactly)
    line = line.lower()
    line = re.sub('[^a-zA-Z_]', ' ', line)
    line = line.strip()
    line = line.replace('  ', '')
    
    prep_text.append(line)

prep = []                  # Removing lenght <=2 and >=15
for line in prep_text:
    temp = ' '
    for word in line.split():
        if len(word) > 2 and len(word) < 15:
            temp += word + ' '
            
    temp = temp.strip()
    prep.append(temp)

"""Creating DataFrame"""

clean_df = pd.DataFrame()
clean_df['text'] = text 
clean_df['class'] = data['class']                  #Stored class in data.csv
clean_df['preprocessed_text'] = prep
clean_df['preprocessed_subject'] = data['preprocessed_subject']
clean_df['preprocessed_email']= preprocessed_email

sub = []
for i in clean_df['preprocessed_subject']:
    i = i.lower()
    i = i.strip()
    sub.append(i)
clean_df['preprocessed_subject'] = sub

clean_df.head(1)

len(clean_df)
def preprocess(Input_Text):
    
    path = 'documents'          # Get index of File
    with os.scandir(path) as dirr:
        idx = 1
        for file in dirr: 
            if file.name == Input_Text :
                break
            else:
                idx += 1
                
    data = pd.read_csv('clean2.csv')
    
    pp_email = (data['preprocessed_email'].iloc[idx])
    pp_subject = (data['preprocessed_subject'].iloc[idx])
    pp_text = (data['preprocessed_text'].iloc[idx])
    
    return pp_email , pp_subject , pp_text
