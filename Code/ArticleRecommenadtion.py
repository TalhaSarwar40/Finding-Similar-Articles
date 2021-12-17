#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 22:48:53 2021

@author: talhasarwar
"""

import pke
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import numpy as np
import math
import spacy
nlp = spacy.load("en_core_web_lg")

#----------------Similarity Calculation Functions start here---------------------------

def cosineSimilarityWord2Vec(articleName, MAKp, Kp):
    
    MAKpList = []
    KpList = []
    
    for j in range(0,len(MAKp),1):
        MAKpList.append(MAKp[j][0])
        
    MAKpString = ((' '. join(MAKpList)))
    #print(MAKpString)
    
    for i in range(0,len(Kp),1):
        KpList.append(Kp[i][0])
        
    KpString = ((' '. join(KpList)))
    
    mainArticle = nlp(MAKpString)
    similarArticle = nlp(KpString)
    
    similarityWord2Vec = mainArticle.similarity(similarArticle)
    #print(similarityWord2Vec)
    sim_Article_Tuple = tuple([articleName.strip(), similarityWord2Vec]) 
    
    return sim_Article_Tuple
    

def jaccardSimilarity(articleName, MAKp, Kp):
    
    #print(MAKp)
    #print(Kp)
    count = 0
    MAKpList = []
    KpList = []
    
    for j in range(0,len(MAKp),1):
        
        main_words = word_tokenize(MAKp[j][0])
        wm = []
        for word in main_words:
            x = ps.stem(word)
            wm.append(x)
            #print(wa)
            #print(words)
        mainKp = (' '. join(wm))
        MAKpList.append(mainKp)
    
    for j in range(0,len(Kp),1):
        
        sim_words = word_tokenize(Kp[j][0])
        ws = []
        for word in sim_words:
            x = ps.stem(word)
            ws.append(x)
            #print(wa)
            #print(words)
        simKp = (' '. join(ws))
        KpList.append(simKp)
    
    #print(MAKpList)
    #print(KpList)
    for i in range(0,len(MAKpList),1): 
        
        for j in range(0,len(KpList),1):
            if MAKpList[i].strip() == KpList[j].strip():
                count += 1
    
    jaccard_similarity = float(count / (len(MAKpList)+len(KpList)-count))
    sim_Article_Tuple = tuple([articleName.strip(), jaccard_similarity])   
        
    return sim_Article_Tuple
    
    
    

def cosineSimilarity(articleName, MAKp, Kp):
    
    list1 = []
    list2 = []
    dot_Product = 0
    sum_sqr_1 = 0
    sum_sqr_2 = 0
    cosine_similarity = 0
    
    #print(MAKp)
       
    for i in range(0,len(MAKp),1):

        list1.append(MAKp[i][1])
        #print(type(MAKp[i][1]))
        list2.append(0)
    #print(articleName)
    
    for i in range(0,len(MAKp),1):
        
        for j in range(0,len(Kp),1):
            
            #main = ps.stem(MAKp[i][0])
            #sim = ps.stem(Kp[j][0])
            
        #stemming starts---------****----------
            main_words = word_tokenize(MAKp[i][0])
            wm = []
            for word in main_words:
                x = ps.stem(word)
                wm.append(x)
                #print(wa)
            #print(words)
            mainKp = (' '. join(wm))
            
            sim_words = word_tokenize(Kp[j][0])
            ws = []
            for word in sim_words:
                x = ps.stem(word)
                ws.append(x)
                #print(wa)
            #print(words)
            simKp = (' '. join(ws))
        #stemming ends---------****----------
            
            if mainKp.strip() == simKp.strip():
                
                list2[i] = float(Kp[j][1])
                #print(list2[i])
    #print("-----------------------")
    #print(list2)
    for j in range(0,len(list1),1):
        
        sum_sqr_1 += (list1[j] * list1[j])
    
    for j in range(0,len(list2),1):
        
        sum_sqr_2 += (list2[j] * list2[j])
    
    dot_Product = np.dot(list1, list2)
    
    norm_a = math.sqrt(sum_sqr_1)
    norm_b = math.sqrt(sum_sqr_2)
    
    if (norm_a * norm_b) != 0:
        
        cosine_similarity = (dot_Product / (norm_a * norm_b))
    else:
        cosine_similarity = 0
        
    sim_Article_Tuple = tuple([articleName.strip(), cosine_similarity])   
    
    
    return sim_Article_Tuple

#----------------Similarity Calculation Functions end here---------------------------
    
#----------------Keyphrase Extraction Functions start here---------------------------
    
    
def getKeyphrasesKPMiner(path, stoplist):
    
    #print(path)
    extractor = pke.unsupervised.KPMiner()

    extractor.load_document(input=path,
                        language='en',
                        normalization=None)
    
    lasf = 5
    cutoff = 200

  
    extractor.candidate_selection(lasf=lasf, cutoff=cutoff)


    # 4. weight the candidates using KPMiner weighting function.
    #df = pke.load_document_frequency_file(input_file='test/1571941.1572113.pdf.txt')
    alpha = 2.3
    sigma = 3.0
    extractor.candidate_weighting(df=None, alpha=alpha, sigma=sigma)

    # 5. get the n-highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=25)
    
    return keyphrases


def getKeyphrasesYAKE(path, stoplist):
    
    #print(path)
    extractor = pke.unsupervised.YAKE()

    extractor.load_document(input=path,
                        language='en',
                        normalization=None)

    extractor.candidate_selection(n=3, stoplist=stoplist)
    window = 2
    use_stems = True # use stems instead of words for weighting
    extractor.candidate_weighting(window=window, stoplist=stoplist, use_stems=use_stems)
    
    #print(extractor.sentences[0].stems)
    threshold = .8
    keyphrases = extractor.get_n_best(n=25, stemming = False, redundancy_removal=True, threshold=threshold)
    
    return keyphrases


def getKeyphrasesMultipartiteRank(path, stoplist):
    
    #print(path)
    extractor = pke.unsupervised.MultipartiteRank()

    extractor.load_document(input=path,
                        language='en',
                        normalization=None)
    
    # define the valid Part-of-Speeches to occur in the graph
    pos = {'NOUN', 'PROPN', 'ADJ'}
    
    #stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    
    extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')
    
    #print(extractor.sentences[0].stems)
    
    keyphrases = extractor.get_n_best(n=25, stemming = False)
    
    return keyphrases


def getKeyphrasesTopicRank(path, stoplist):
    
    #print(path)
    extractor = pke.unsupervised.TopicRank()

    extractor.load_document(input=path,
                        language='en',
                        normalization=None)
    
    pos = {'NOUN', 'PROPN', 'ADJ'}

    extractor.candidate_selection(stoplist=stoplist, pos = pos)
    
    extractor.candidate_weighting(threshold=0.74, method='average')
    
    #print(extractor.sentences[0].stems)
    
    keyphrases = extractor.get_n_best(n=25, stemming = False)
    
    return keyphrases


def getKeyphrasesKea(path, stoplist):
    
    '''
    print(path)
    #print("-------------------In Kea Function----------------------")
    f = open(path, "r")
    text = f.read()
    #print(text)
    '''
    # 1. create a Kea extractor.
    extractor = pke.supervised.Kea()

    # 2. load the content of the document.
    extractor.load_document(input=path,
                            language='en',
                            normalization=None)

    # 3. select 1-3 grams that do not start or end with a stopword as
    #    candidates. Candidates that contain punctuation marks as words
    #    are discarded.
    extractor.candidate_selection(stoplist=stoplist)

    # 4. classify candidates as keyphrase or not keyphrase.
    df = pke.load_document_frequency_file(input_file='model/df-semeval2010.tsv.gz')
    model_file = 'model/Kea-semeval2010.py3.pickle'
    extractor.candidate_weighting(model_file=model_file, df=df)
                                  

    # 5. get the 10-highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=25)
    
    return keyphrases


#----------------Keyphrase Extraction Functions end here---------------------------


#---------------------------------------Main program starts here------------------------------------------------------------

stoplist = stopwords.words('english')
simCosineList = [] 
simJaccardList = []
simWord2VecList = []

#----------------root nenws article's keyphrase extraction starts------------------------------------------------
main_article_path = "root/It's not just Delta -- other coronavirus variants worry scientists, also.txt"
#print(main_article_path)
main_path_split = main_article_path.split("/")      
main_paper_name = ''.join(main_path_split[1]) 

#main_article_keyphrase = getKeyphrasesKPMiner(main_article_path, stoplist)
main_article_keyphrase = getKeyphrasesYAKE(main_article_path, stoplist)
#main_article_keyphrase = getKeyphrasesMultipartiteRank(main_article_path, stoplist)
#main_article_keyphrase = getKeyphrasesTopicRank(main_article_path, stoplist)
#main_article_keyphrase = getKeyphrasesKea(main_article_path, stoplist)
print("Main Article Keyphrases: ") 
#print(main_article_keyphrase) 
list2dict = dict(main_article_keyphrase) 
print(list2dict)

KP = []
for i in range(0,len(main_article_keyphrase), 1):
    KP.append(main_article_keyphrase[i][0])
    #print(main_article_keyphrase[i][0])
#----------------root nenws article's keyphrase extraction ends----------------------
#print(KP)

#--------------------recommended nenws article's keyphrase extraction starts here--------------------------------
    
full_path = [os.path.join(r,file) for r,d,f in os.walk("Data") for file in f]

print("Total Documents:" + str(len(full_path)))

for i in range(0,len(full_path),1):
    
    #print(i)
    #print(full_path[i])
    
    path_split = full_path[i].split("/")
        
    paper_name = ''.join(path_split[1]) 
    
    #print(paper_name)
    
    #print(full_path[i])
    
#-------------------------------Keyphrase extraction function calling---------------------------
    
    #keyphrases = getKeyphrasesKPMiner(full_path[i], stoplist) #----KP-Miner----
    #keyphrases = getKeyphrasesYAKE(full_path[i], stoplist)  #----YAKE----
    #keyphrases = getKeyphrasesMultipartiteRank(full_path[i], stoplist)  #----Multipartite----
    #keyphrases = getKeyphrasesTopicRank(full_path[i], stoplist) #----TopicRak----
    #keyphrases = getKeyphrasesKea(full_path[i], stoplist)   #----Kea----
    #print(keyphrases)
    #print("--------------------------")
    
    
#-------------------------------Similarity Calculation function calling starts here-----------------------------------------------------
    
    #news_article_similarity_tuple = cosineSimilarity(paper_name, main_article_keyphrase, keyphrases) #calculate cosine similarity
    #simCosineList.append(news_article_similarity_tuple)
    
    #news_article_similarity_tuple = jaccardSimilarity(paper_name, main_article_keyphrase, keyphrases) #calculate jaccard similarity
    #simJaccardList.append(news_article_similarity_tuple)
    
    #news_article_similarity_tuple = cosineSimilarityWord2Vec(paper_name, main_article_keyphrase, keyphrases)
    #simWord2VecList.append(news_article_similarity_tuple)
    
#-------------------------------Similarity Calculation function calling ends here--------------------------------------------------------
    
    '''
    print("Main Article Name: " + str(main_paper_name))
    print("Similar Article Name: " + str(news_article_similarity_tuple[0]))
    print("Similarity Score: " + str(news_article_similarity_tuple[1]))
    
    #print(news_article_similarity_tuple)
    print("---------------------------")
    '''
    
#--------------------recommended nenws article's keyphrase extraction ends here-----------------------
print("")
#print(simCosineList)


#-----------------Sorting Lists----------------------
simCosineList.sort(key=lambda y: y[1], reverse = True)
simJaccardList.sort(key=lambda y: y[1], reverse = True)
simWord2VecList.sort(key=lambda y: y[1], reverse = True)

#print(type(simCosineList))

#print(simCosineList)


'''
#----------------print results of cosine similarity---------------
for j in range(0,len(simCosineList),1):
    
     
    print("Main Article Name: " + str(main_paper_name))
    print("Similar Article Name: " + str(simCosineList[j][0]))
    print("Cosine Similarity Score: " + str(simCosineList[j][1]))
    print("---------------------------")
    
#----------------end print results of cosine similarity---------------   


#----------------print results of jaccard similarity---------------
for j in range(0,len(simJaccardList),1):
    
     
    print("Main Article Name: " + str(main_paper_name))
    print("Similar Article Name: " + str(simJaccardList[j][0]))
    print("Jaccard Similarity Score: " + str(simJaccardList[j][1]))
    print("---------------------------")
    
#----------------end print results of jaccard similarity--------------- 

#----------------print results of Word2Vec similarity---------------
for j in range(0,len(simWord2VecList),1):
    
     
    print("Main Article Name: " + str(main_paper_name))
    print("Similar Article Name: " + str(simWord2VecList[j][0]))
    print("Word2Vec Similarity Score: " + str(simWord2VecList[j][1]))
    print("---------------------------")
    
#----------------end print results of Word2Vec similarity--------------- 
'''    
    
    
    
    
