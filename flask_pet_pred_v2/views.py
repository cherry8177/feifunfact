from flask import render_template
from flask_pet_pred_v2 import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import pickle
import numpy as np
from flask import request
from flask_pet_pred_v2.data_transform import data_clean, data_vectorize, data_predict, get_prob
from flask_pet_pred_v2.plotly import plotly_prob
import pickle
import random

# user = 'feiwang' #add your username here (same as previous postgreSQL)            
# host = 'localhost'
# dbname = 'birth_db'
# db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

# # @app.route('/')
# @app.route('/index')
# def index():
#     return render_template("index.html",
#        title = 'Home', user = { 'nickname': 'Miguel' },
#        )

# @app.route('/db')
# def birth_page():
#     sql_query = """                                                             
#                 SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\;                                                                               
#                 """
#     query_results = pd.read_sql_query(sql_query,con)
#     births = ""
#     print( query_results[:10])
#     for i in range(0,10):
#         births += query_results.iloc[i]['birth_month']
#         births += "<br>"
#     return births

# @app.route('/db_fancy')
# def cesareans_page_fancy():
#     sql_query = """
#                SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
#                 """
#     query_results=pd.read_sql_query(sql_query,con)
#     births = []
#     for i in range(0,query_results.shape[0]):
#         births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
#     return render_template('cesareans.html',births=births)
@app.route('/')
@app.route('/about')
def petition_about():
  #  r = request.forms['birth_month']
    return render_template("about1.html")


@app.route('/input')
def petition_input():
  #  r = request.forms['birth_month']
    return render_template("input.html")

@app.route('/output' , methods=['GET', 'POST'])
def petition_output():
    if request.args.get('Preload')=='True':
        with open('flask_pet_pred_v2/sample_df.pickle','rb') as f:
            sample_df= pickle.load(f)

        n=random.randint(0,10)
        Title1 = sample_df.Title[n]
        Text1 = sample_df.Text[n]
        return render_template("input_preload.html",  Title1=Title1, Text1=Text1)
    else:

        Title1 = request.args.get('Title')
        Text1 = request.args.get('Text')
        Image1 = request.args.get('Image')=='True'
        Tweet1 = request.args.get('Tweet')=='True'

    Goal_No1 = request.args.get('Goal_No')

    try:
        val=int(Goal_No1)
    except ValueError:
        Goal_No1=1000

    fields=[Title1, Text1, Goal_No1]
    

    if not all(fields):
    	
    	return render_template("re_input.html", Title1=Title1, Text1=Text1)

    df=data_clean(Title1,Text1,Goal_No1,Image1,Tweet1)

    if len(df.Title.str.split(" ")[0])<3 or len(df.Text.str.split(" ")[0])<15:
    	
    	return render_template("re_input.html")

    x_test_pca=data_vectorize(df)
    y_predict=data_predict(x_test_pca)
    the_result=int(10**y_predict)
    log_goal=float(np.log10(df.Goal_No))
    prob=get_prob(log_goal, y_predict)
    prob= round(prob,3)*100
   
    plotly_fig=plotly_prob(y_predict, log_goal)





    #just select the Cesareans  from the birth dtabase for the month that the user inputs
    # query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
    # print(query)
    # query_results=pd.read_sql_query(query,con)
    # print(query_results)
    # births = []
    # for i in range(0,query_results.shape[0]):
    #     births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    #     the_result = ModelIt(patient,births)
    return render_template("output.html", 
        Title1=Title1,Text1=Text1,Goal_No1=Goal_No1, 
        plotly_fig=plotly_fig,prob=prob, Title=Title1,the_result = the_result)
