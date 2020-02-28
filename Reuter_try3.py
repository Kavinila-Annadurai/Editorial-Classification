import os
import csv
import pandas as pd
import matplotlib.pyplot as plt        
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection,naive_bayes,svm
from sklearn.metrics import accuracy_score,f1_score

#path = r"C:\Users\hp\Downloads\Reuters21578-Apte-90Cat\training"
#entries = os.listdir(path)
#    
#with open("file.csv", 'w') as csvfile: 
#    # creating a csv writer object 
#    csvwriter = csv.writer(csvfile,dialect="excel") 
#    # writing the fields 
#    csvwriter.writerow(["Topic","Document"])     
#    # writing the data rows
#    for i in range(len(entries)):#len(entries)
#        entry = os.listdir(path + "\\" + entries[i]) 
#        for j in range(len(entry)):
#            f=open(r"C:\Users\hp\Downloads\Reuters21578-Apte-90Cat\training"+"\\" + entries[i]+"\\" + entry[j], "r")
#            contents = f.read()
#            app=[]
#            app.append(entries[i])
#            app.append(contents)
#            #print(app[1])
#            csvwriter.writerow(app)
#            f.close()
#csvfile.close()

data = pd.read_csv(r'file.csv',encoding = 'latin-1') 

#Pre-processing steps

#Remove blank lines if any
data.dropna(inplace = True)

#Convert upper case to lower
data['Document'] = data['Document'].str.lower() 

#Tokenization
data['Document'] = [ word_tokenize(entry) for entry in data['Document'] ]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(data['Document']):
    final_words = []
    word_lem = WordNetLemmatizer()
    for word,tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            f_word = word_lem.lemmatize(word,tag_map[tag[0]])
            final_words.append(f_word)
    data.loc[index,'text_final'] = str(final_words)
    

    
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)

Tfidf_vect = TfidfVectorizer(max_features=None)
Tfidf_vect.fit(data['text_final'])

TP_nb = 0
TP_svm = 0 

nb_accuracy = 0
svm_accuracy = 0

for train, test in kf.split(data['text_final']):
    
    X_train = data['text_final'][train]
    X_test = data['text_final'][test]
    y_train = data['Topic'][train]
    y_test = data['Topic'][test]
    l = len(X_test)
    

    X_train_final = Tfidf_vect.transform(X_train).toarray()
    X_test_final = Tfidf_vect.transform(X_test).toarray()
    
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(X_train_final,y_train)
    
    pred = Naive.predict(X_test_final)
    nb_accuracy+=accuracy_score(pred,y_test)
    CM = confusion_matrix(y_test, pred)
    TP_nb += CM[1][1]
    
    
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(X_train_final,y_train)
    pred = SVM.predict(X_test_final)
    svm_accuracy+=accuracy_score(pred,y_test)
    CM = confusion_matrix(y_test, pred)
    TP_svm += CM[1][1]
    
TP_nb/=(kf.get_n_splits(data['text_final']))
TP_svm/=(kf.get_n_splits(data['text_final']))

svm_accuracy /= (kf.get_n_splits(data['text_final']))
nb_accuracy /= (kf.get_n_splits(data['text_final'])) 

print("Naive Bayes Accuracy Score:",nb_accuracy)
print("SVM Accuracy Score:",svm_accuracy)


from scipy import stats
import numpy as np
p_cap1 = (TP_nb/l)
p_cap2 = (TP_svm/l)
 
p_cap = ((TP_nb+TP_svm)/(2*l))
Z = ((p_cap1-p_cap2)/np.sqrt(2*p_cap*(1-p_cap)/l))

p_value = stats.norm.pdf(abs(Z))*2
print(Z," ",p_value)



