from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from clear_data import get_sentence
from symmetric_similarity import *
from preproccessing import tfidf
import csv
import io

count_questions = 20000 # the number of question pairs
questions = [] # for appending the two questions
is_duplicate = [] # for computing accuracy later
count_duplicates = 0 # for computing accuracy later

# Read & process our dataset
with io.open('train_original.csv', mode = 'r' , encoding='utf8') as csvfile:
    reader = csv.DictReader(csvfile)
    i = 0
    for row in reader:
        i += 1
        questions.append(get_sentence(row['question1']))
        questions.append(get_sentence(row['question2']))
        is_duplicate.append(int(row['is_duplicate']))
        if (int(row['is_duplicate'])):
            count_duplicates += 1
        if i >= count_questions:
            break

# Statistic Model
ftidf_vectorizer = TfidfVectorizer()
questions_ftidf = ftidf_vectorizer.fit_transform(questions)

ftidf_similarity = []
for i in range(0,count_questions,2):
    ftidf_similarity.append(cosine_similarity(questions_ftidf[i], questions_ftidf[i+1]))

# Semantic Model
symmetric_similarity = []
for i in range(0,count_questions,2):
    symmetric_similarity.append(symmetric_question_similarity(questions[i], questions[i+1]))


"""
Semantic Model & Statistic Model Accuracy

By changing the statistic_perc and semantic_perc variables
you set the percentage that each model takes part in the final prediction.
In the example below I only used the Semantic Model.
"""
index =0
correct_ans = 0

# These two must sum up to 1.0
statistic_portion = 0.0
semantic_portion = 1.0

threshold = 0.9

for i in range(int(count_questions/2)):
    similarity = (statistic_portion*ftidf_similarity[i] + semantic_portion*symmetric_similarity[i])
    if similarity > threshold:
        label = 1
    else:
        label = 0
    if label == is_duplicate[index]:
        correct_ans +=1
    index += 1

print(correct_ans/count_questions*2)