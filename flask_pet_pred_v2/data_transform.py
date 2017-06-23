
import pandas as pd
import pickle
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
import os
import numpy as np
import scipy.stats

Title = "this is the title"
Text = "I am the text. Bitch!"
Image = 0
Tweet = 1
Goal_No = '2500'

def data_clean(Title1,Text1,Goal_No1,Image1,Tweet1):
    
    def str2float(s):
        s = str(s).strip().replace(',','')
        s = re.sub('[^0-9]+', '0', s)
        return float(s)
    
    def remove_sw(words, sw):
        word = [w for w in words if not w in sw]
        return word
    
    def clean_text( col):
        letters_only=(col.apply(lambda x:re.sub(u"\xa0",u" ",x))
        #.apply(lambda x: BeautifulSoup(x).get_text())
        .apply(lambda x:re.sub("[^a-zA-Z]"," ",x))
                 )
        lower_case=letters_only.apply(lambda x: x.lower().split())
        from nltk.corpus import stopwords # Import the stop word list
        stopwords=set(stopwords.words("english"))
        clean_texts = []
        num_texts = col.size
        for i in range( 0, num_texts ):
        # Call our function for each one, and add the result to the list of
        # clean 
            clean_texts.append( " ".join(remove_sw(lower_case[i],stopwords)))
        return clean_texts
    

    raw_dict={'Title':[Title1],'Text':[Text1],'Goal_No':[Goal_No1],'Image':Image1,'Tweet':Tweet1}
    df=pd.DataFrame(raw_dict)
    df.Goal_No=df.Goal_No.apply(str2float)
    df.Text=clean_text(df.Text)
    df.Title=clean_text(df.Title)
    df['Text_len']=df.Text.str.split(' ').apply(lambda x: len(x))
    df['Title_len']=df.Title.str.split(' ').apply(lambda x: len(x))
    df['Text_str_len']=df.Text.str.split(' ').apply(lambda x: sum(len(w) for w in x)/ len(x))
    df['Title_str_len']=df.Title.str.split(' ').apply(lambda x: sum(len(w) for w in x)/ len(x))
    df['Text_len_p1']=(df['Text_len']<270)*df['Text_len']
    df['Text_len_p2']=(df['Text_len']>270)*df['Text_len']
    return df
    
    
    

    

def data_vectorize(df):
    

    with open('flask_pet_pred_v2/text_court_vectorizer_v2.pickle','rb') as f:
        text_court_vectorizer=pickle.load(f)
    with open('flask_pet_pred_v2/title_court_vectorizer_v2.pickle','rb') as f:
        title_court_vectorizer=pickle.load(f)
    with open('flask_pet_pred_v2/total_scale_train_v2.pickle','rb') as f:
        total_scale_train=pickle.load(f)
    with open('flask_pet_pred_v2/train_pca.pickle_v2','rb') as f:
        train_pca=pickle.load(f)
    
    
    test_text_features = text_court_vectorizer.transform(df.Text)
    test_text_features = test_text_features.toarray()  
    test_title_features = title_court_vectorizer.transform(df.Title)
    test_title_features = test_title_features.toarray()
    

    test_all_features=total_scale_train.transform(np.concatenate((
                        test_text_features,test_title_features,
                        df[['Title_str_len', 'Text_len_p1', 'Text_len_p2', 
                        'Title_len', 'Text_str_len','Image','Tweet']]),axis=1))
    
    
    x_test_pca=train_pca.transform(test_all_features)
    
    return x_test_pca
    


def data_predict(x_test_pca):
    with open('flask_pet_pred_v2/train_Elastic_v2.pickle','rb') as f:
        train_Elastic=pickle.load(f)
    test_predict=train_Elastic.predict(x_test_pca[:,0:1000])
    return test_predict


def get_prob(goal_No, y_predict):

    prob=scipy.stats.norm(goal_No,0.71**0.5).cdf(y_predict)
    return prob[0]


    
    

