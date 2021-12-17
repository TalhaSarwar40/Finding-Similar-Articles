
#from scikit-learn import sklearn
from sklearn.metrics import ndcg_score, dcg_score
import numpy as np
  
# Relevance scores in Ideal order
true_relevance = np.asarray([[2,2,1,1,0]])
  
# Relevance scores in output order
relevance_score = np.asarray([[1, 2, 2, 0, 1]])
  
# DCG score
dcg = dcg_score(true_relevance, relevance_score)
print("DCG score: ", dcg)
  
# IDCG score
idcg = dcg_score(true_relevance, true_relevance)
print("IDCG score: ", idcg)
  
# Normalized DCG score
ndcg = dcg / idcg
print("nDCG score: ", ndcg)
  
# or we can use the scikit-learn ndcg_score package
print("nDCG score (from function): ", ndcg_score(
    true_relevance, relevance_score))


'''
import numpy as np
from sklearn.metrics import dcg_score
# we have groud-truth relevance of some answers to a query:
true_relevance = np.asarray([[10, 0, 0, 1, 5]])
# we predict scores for the answers
scores = np.asarray([[.1, .2, .3, 4, 70]])
dcg_score(true_relevance, scores)

# we can set k to truncate the sum; only top k answers contribute
dcg_score(true_relevance, scores, k=2)

# now we have some ties in our prediction
scores = np.asarray([[1, 0, 0, 0, 1]])
# by default ties are averaged, so here we get the average true
# relevance of our top predictions: (10 + 5) / 2 = 7.5
dcg_score(true_relevance, scores, k=1)

# we can choose to ignore ties for faster results, but only
# if we know there aren't ties in our scores, otherwise we get
# wrong results:
dcg_score(true_relevance, scores, k=1, ignore_ties=True)
'''