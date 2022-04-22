#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# # Importing Data

# In[16]:


data = pd.read_csv(r"C:\Users\serka\Desktop\YTÜ\YTÜ 3-2\proje\notCodeSmell.csv")


# In[17]:


data


# ## Removing Missing Values

# In[18]:


data = data.dropna()


# ## Removing Duplicates, Effort Calculation With COCOMO and  Our Formula

# In[19]:


data_clean = data.drop_duplicates()
data_clean = data_clean.reset_index(drop=True)
sum_lines = data_clean["linesAdded"]+data_clean["linesRemoved"]
data_clean["KLOC_EFOR"]=sum_lines
sum_lines = 2.4*data_clean["KLOC_EFOR"]**1.05
data_clean["KLOC_EFOR"] = sum_lines

sum_lines = data_clean["linesAdded"]+data_clean["linesRemoved"]
data_clean["ESTIMATED_EFOR"]=sum_lines
sum_lines = data_clean["ESTIMATED_EFOR"]/data_clean["nloc"]
data_clean["ESTIMATED_EFOR"]=sum_lines
sum_lines = data_clean["ESTIMATED_EFOR"]*data_clean["complexity"]
data_clean["ESTIMATED_EFOR"]=sum_lines

sum_lines = data_clean["newSqaleDebtRatio"]*data_clean["ESTIMATED_EFOR"]*7
data_clean["ESTIMATED_EFOR"]=sum_lines

data_clean


# ## Exporting New Dataset to .CSV

# In[8]:


data_clean.to_csv("clean_data.csv")


# ## Group By JIRA_KEY and Sum of Real and Estimated Effort

# In[10]:


data_grouped = data_clean.groupby(["JIRA_KEY"])["KLOC_EFOR","effort","ESTIMATED_EFOR"].sum().reset_index()


# In[11]:


data_grouped


# ## Adding Sonar Metrics to Grouped Data

# In[12]:


df= pd.DataFrame()
for i in range(207):
    data = pd.read_csv(r"C:\Users\serka\Desktop\YTÜ\YTÜ 3-2\proje\clean_data.csv")
    jira_key = data_grouped["JIRA_KEY"][i]
    filter1 = data["JIRA_KEY"]==jira_key
    data.where(filter1,inplace=True)
    data=data.dropna()
    data = data.reset_index(drop=True)
    data = data.drop_duplicates("JIRA_KEY")
    df=df.append(data)
df = df.reset_index(drop=True)
df_merged = pd.merge(data_grouped,df,on = "JIRA_KEY")
df_merged = df_merged.drop(["key","sqaleRating"],axis=1)


# In[13]:


df_merged.reset_index(drop=True)


# In[14]:


df_merged = df_merged.drop(["Unnamed: 0", "KLOC_EFOR_y","ESTIMATED_EFOR_y","faultFixingCommitHash"],axis=1)


# ## Exporting Merged and Final Dataset to .CSV 

# In[15]:


df_merged.to_csv("notCodeSmell_merged.csv")


# ## Hata_Kaydı Embeddings

# In[22]:


text_data = KLOC_data_clean["HATA_KAYDI_OZET"]
desc_data = KLOC_data_clean["description"]


# In[23]:


text_data = text_data.drop_duplicates()
desc_data = desc_data.drop_duplicates()


# In[24]:


text_data = text_data.reset_index(drop=True)
desc_data = desc_data.reset_index(drop=True)


# In[25]:


text_data = text_data.tolist()
desc_data = desc_data.tolist()


# In[26]:


text_data


# In[74]:


from sentence_transformers import SentenceTransformer
model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)
sentence_vecs = model.encode(text_data)
sentence_vecs


# In[78]:


k = 0
df = []
for i in range (0,7844):
    if text_data[k] == KLOC_data_clean["HATA_KAYDI_OZET"][i]:
        df.append(sentence_vecs[k])
    else:
        if k+1 < 1655:
            k = k+1
        df.append(sentence_vecs[k])


# In[79]:


KLOC_data_clean_0_7["embeddings_ozet"] = df
KLOC_data_clean_0_75["embeddings_ozet"] = df
KLOC_data_clean_0_8["embeddings_ozet"] = df
KLOC_data_clean_0_85["embeddings_ozet"] = df
KLOC_data_clean_0_9["embeddings_ozet"] = df


# In[80]:


model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)
sentence_vecs = model.encode(desc_data)
sentence_vecs


# In[81]:


k = 0
df = []
for i in range (0,7844):
    if desc_data[k] == KLOC_data_clean["description"][i]:
        df.append(sentence_vecs[k])
    else:
        if k+1 < 1655:
            k = k+1
        df.append(sentence_vecs[k])


# In[82]:


KLOC_data_clean_0_7["embeddings_desc"] = df
KLOC_data_clean_0_75["embeddings_desc"] = df
KLOC_data_clean_0_8["embeddings_desc"] = df
KLOC_data_clean_0_85["embeddings_desc"] = df
KLOC_data_clean_0_9["embeddings_desc"] = df


# In[27]:


from tensorflow import keras
import pandas as pd
import numpy as np


# In[28]:


from bpemb import BPEmb


# In[29]:


bpemb_en = BPEmb(lang="en", vs=50000, dim=25)


# In[30]:


sentences = np.array([np.array(bpemb_en.embed(x)) for x in desc_data])


# In[31]:


rnn = keras.layers.SimpleRNN(3, kernel_initializer=keras.initializers.ones, recurrent_initializer=keras.initializers.zeros, activation="tanh")


# In[32]:


max_len = max(map(len, sentences))


# In[33]:


rnn.build(input_shape=(1,max_len,25))


# In[34]:


import tensorflow as tf


# In[35]:


def calculate_distances(sentences):
    values = {}
    for i in range(len(sentences)):
        # Set the weights manually, so the length of the sentence will be concerned
        rnn.set_weights([rnn.get_weights()[0], tf.constant(1/len(sentences[i]),shape=(3,3)), tf.constant(1/len(sentences[i]),shape=(3))])
        values[desc_data[i]] = rnn(np.array([sentences[i]]))
    # Calculate a simple absolute distance, you might want to use another metric for this
    distances = [[np.absolute((l-i2)[0][0]) for i2 in values.values()] for l in values.values()]
    df = pd.DataFrame(index=values.keys(), columns=values.keys(), data=distances)
    return df


# In[36]:


calculate_distances(sentences)


# ## En yakın hata mesajı bulunması

# In[ ]:





# ## Cosine 

# In[46]:


from sklearn.metrics.pairwise import cosine_similarity


# In[80]:


deneme = KLOC_data_clean_0_7["embeddings"].to_numpy()


# In[84]:


deneme


# In[81]:


deneme[1].shape


# In[82]:


deneme.shape


# In[106]:


uzaklık = cosine_similarity([sentence_vecs[0]],sentence_vecs[1:])


# In[111]:


type(uzaklık)


# In[116]:


uzaklık[0][]


# In[119]:


k=0
for i in uzaklık[0]:
    print(k)
    print(i)
    
    k += 1


# In[115]:


index = np.where(uzaklık[0] == 0.739)
index


# In[ ]:




