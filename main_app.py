from flask import Flask, render_template, request
#import re
import pandas as pd
#import copy
import pickle
import joblib

model_rf = pickle.load(open('RF.pkl','rb'))
model_knn = pickle.load(open('KNN.pkl', 'rb'))
#model_dt = pickle.load(open('DT.pkl','rb'))
imp_enc_scale = joblib.load('Num_CatPipeline')
winsor = joblib.load('winsor')

from sqlalchemy import create_engine
from urllib.parse import quote 
user_name = 'root' #userid
database = 'mobile_db' # databasename
your_password = 'Frekey01$' #password
engine = create_engine(f'mysql+pymysql://{user_name}:%s@localhost:3306/{database}' % quote(f'{your_password}'))

def decision_tree(data_new):
    clean1 = pd.DataFrame(imp_enc_scale.transform(data_new), columns = imp_enc_scale.get_feature_names_out())
    clean1[['num__Battery_Power', 'num__Clock_Speed', 'num__FC', 'num__Int_Memory', 'num__Mobile_D', 'num__Mobile_W', 
              'num__Cores', 'num__PC', 'num__Pixel_H', 'num__Pixel_W', 'num__Ram', 'num__Screen_H', 'num__Screen_W', 'num__Talk_Time']] = winsor.transform(clean1[['num__Battery_Power', 'num__Clock_Speed', 'num__FC', 'num__Int_Memory', 'num__Mobile_D', 'num__Mobile_W', 
                        'num__Cores', 'num__PC', 'num__Pixel_H', 'num__Pixel_W', 'num__Ram', 'num__Screen_H', 'num__Screen_W', 'num__Talk_Time']])
    #prediction = pd.DataFrame(model_dt.predict(clean1), columns = ['DT_Price_Range'])
    prediction = pd.DataFrame(model_knn.predict(clean1), columns = ['KNN'])
    prediction2 = pd.DataFrame(model_rf.predict(clean1), columns = ['RF_Price'])                      
    final_data = pd.concat([prediction, prediction2, data_new], axis = 1)
    return(final_data)
            
#define flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data_new = pd.read_csv(f)
       
        final_data = decision_tree(data_new)
        
        final_data.to_sql('Mobile_Data_Prediction'.lower(), con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        
        html_table = final_data.to_html(classes='table table-striped')
                          
        return render_template("new.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #8f6b39;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #32b8b8;\
                    }}\
                            .table tbody th {{\
                            background-color: #3f398f;\
                        }}\
                </style>\
                {html_table}") 

if __name__=='__main__':
    app.run(debug = True)
