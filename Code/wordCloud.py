#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 00:28:28 2021

@author: talhasarwar
"""

import matplotlib.pyplot as plt
#%matplotlib inline
from wordcloud import WordCloud
from matplotlib.pyplot import figure
import pandas as pd
plt.rcParams["figure.figsize"] = (10,10)
# plt.figure(figsize = (8, 😎, facecolor = None)
'''
df = pd.read_csv('cloudKEA.csv')
df.head()
text2 = " ".join(title for title in df.kw)
'''

text = {'variant': 40, 'gamma': 28, 'cnn': 12, 'accord': 12, 'moor': 12}
        
        
word_cloud2 = WordCloud(width = 800, height = 800,colormap = 'plasma', collocations = True, background_color = 'white').fit_words(text)

# Display the generated Word Cloud
plt.figure(facecolor = None, dpi=100)
plt.imshow(word_cloud2, interpolation='bilinear')

plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig("TeKET.pdf")
plt.show()

