
# coding: utf-8

# In[1]:

import graphlab as gl
import pyLDAvis
import pyLDAvis.graphlab
import pandas as pd
import ftfy
import numpy as np
import io


# In[2]:

sf_LVRevJoin = gl.SFrame.read_csv('/home/ubuntu/LARA_Python/data/LVInspRevJoin_Clean_Deduped.csv')
sf_LVRevJoin.head(5)


# In[21]:

gl.canvas.set_target('ipynb')
#sf_train['inspection_grade'].show()
stopwords = gl.text_analytics.stopwords()
stopwords.add("it's")
stopwords.add("-")
stopwords.add("&")
stopwords.add("--")
stopwords.add("2")
stopwords.add("i'm")
stopwords.add("it")
stopwords.add("vegas.")
stopwords.add("vegas")
stopwords.add("las")
stopwords.add(".")
stopwords.add("i've")
stopwords.add("5")

sf_LVRevJoinClean = sf_LVRevJoin.dropna()


# In[4]:

sf_LVRevJoinClean.head(5)


# In[22]:

train_docs = gl.text_analytics.count_words(sf_LVRevJoinClean['rev_text'])
train_docs = train_docs.dict_trim_by_keys(stopwords, exclude=True)


# In[ ]:

topic_model = gl.topic_model.create(train_docs, num_topics=6, num_iterations=10000)


# In[7]:

print topic_model.get_topics().print_rows(num_rows=100, num_columns=10)


# In[8]:

print topic_model.get_topics(output_type='topic_words')


# In[9]:

topic_model.get_topics(num_words=25, output_type='topic_words')['words'][0]


# In[10]:

topic_model.get_topics(num_words=25, output_type='topic_words')['words'][1]


# In[11]:

topic_model.get_topics(num_words=25, output_type='topic_words')['words'][2]


# In[12]:

topic_model.get_topics(num_words=25, output_type='topic_words')['words'][3]


# In[13]:

topic_model.get_topics(num_words=25, output_type='topic_words')['words'][4]


# In[14]:

topic_model.get_topics(num_words=25, output_type='topic_words')['words'][5]


# In[15]:

topic_model.get_topics(num_words=25, output_type='topic_words')['words'][6]


# In[16]:

topic_model.get_topics(num_words=25, output_type='topic_words')['words'][7]


# In[17]:

# turn on automatic rendering of visualizations
pyLDAvis.enable_notebook()


# In[18]:

topic_model_vis = pyLDAvis.graphlab.prepare(topic_model, train_docs)


# In[20]:

pyLDAvis.display(topic_model_vis)


# In[ ]:

pyLDAvis.save_html(topic_model_vis, '/Users/eric/MCSDS/CS498/Project/data/LDAModel.html')


# In[66]:

## import and wrangle restaurant inspections for matching with Yelp Reviews
### import
sfInsp = gl.SFrame('/Users/eric/MCSDS/CS498/Project/data/Restaurant_Inspections.csv')


# In[67]:

sfInsp.head(5)


# In[74]:

### we're going to iterate through these to assign prior and next inspection dates
### so sort by name, address, and date
sfInsp = sfInsp.sort([('Restaurant Name', True) , ('Address', True) , ('Inspection Date', False)])


# In[81]:

### convert to a pandas dataframe so we can iterate
dfInsp = sfInsp.to_dataframe()


# In[90]:

### iterate

    #### walk through each row in the dataframe and set the Prior Inspection Date and Next Inspection Date fields
    #### if this is the first inspection for a restaurant, then the Prior Inspection Date is the Current Inspection Date
    #### if this is the last inspection for a restaurant, then the Next Inspection Date is the Current Inspection Date

PermitNumber = ''

for i in dfInsp.index:

    #print dfInsp.loc[i,'Permit Number']
        
    #### for the first row, or when we switch to a new permit number / restaurant
    #### we don't need to really do anything... set the PermitNumber and the PriorInspDate fields
    
    if PermitNumber != dfInsp.loc[i,'Permit Number']: 
        PriorInspDate = dfInsp.loc[i,'Inspection Date']    
        PermitNumber = dfInsp.loc[i,'Permit Number']
    
    
    #### otherwise, we want to set some fields:
    #### in the current row of the data frame, set the Prior Insp Date field to PriorInspDate
    
    else:
        dfInsp.loc[i,'Prior Insp Date'] = PriorInspDate
    
        #### in the prior row of the data frame, set the Next Insp Date field to the current row's Inspection Date
        dfInsp.loc[i-1, 'Next Insp Date'] = dfInsp.loc[i, 'Inspection Date']


# In[97]:

dfInsp.head(5)


# In[102]:

### get a subset of the fields
dfInsp1 = dfInsp[['Serial Number', 'Permit Number', 'Restaurant Name', 'Location Name', 'Address', 'City', 'State', 'Zip', 'Current Grade', 'Inspection Grade', 'Inspection Date', 'Prior Insp Date', 'Next Insp Date', 'Location 1']]


# In[104]:

### export to csv for further processing
dfInsp1.to_csv('/Users/eric/MCSDS/CS498/Project/data/LVInsp.csv')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



