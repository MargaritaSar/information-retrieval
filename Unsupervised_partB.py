from preprocess import tfidf
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


"""
This is an unsupervised method to check if two sentences are similar.

Find tfidf for each sentence then calculate the cosine similarity for sentences that are in the same line. If the similarity
is bigger than the threshold mark the two sentence as similar otherwise they're not similar
"""

threshold = 0.85

df = pd.read_csv("proccessed_ready.csv")
labels = df['is_duplicate']

#calculate tfidf for each sentence
questions = tfidf()

size = questions.shape[0]
total = int(size/2)

#Label if two sentences are similar or not
correct_ans = 0
for i in range(total):
    similarity = cosine_similarity(questions[i], questions[total+i])
    if similarity >= threshold:
        label = 1
    else:
        label = 0
    #check if the label given from the model is correct
    if label == labels[i]:
        correct_ans += 1

#Calculate the accuracy of the model
print("Accuracy: {}".format(correct_ans/total*100))