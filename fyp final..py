# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:46:48 2019

@author: shawa
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:24:44 2019

@author: shawa
"""


# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import pandas.io.sql
import pyodbc
from apyori import apriori  


global pred,df,server,db,UID,conn
b2=list()
row2=list()
server = 'DESKTOP-HOCP62I\SQLEXPRESS'
db = 'SmartMart'
UID = 'shawaizhafeez@live.com'
connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=DESKTOP-HOCP62I\SQLEXPRESS;DATABASE=SmartMart;Trusted_Connection=yes')

#sql= """
#
#SELECT * FROM OnlineRetailDb
#
#"""
#df=pd.read_sql(sql, conn)
#print("hi ai am working")
#print(df.head())
def sqlEntryRfm(d,connStr):
    cursor = connStr.cursor()
    print("starting to enter database")    
    for index,row in d.iterrows():
        cursor.execute("INSERT INTO dbo.Rfm([CustomerID],[Recency_Flag],[Freq_Flag],[Monetory_Flag],[Overall_Percentage]) values (?, ?,?,?,?)", row['CustomerID'], row['Recency_Flag'] , row['Freq_Flag'],row['Monetory_Flag'],row['Overall_Percentage'])
        connStr.commit()
    cursor.close()
    #connStr.close()

def sqlEntryClv(d,connStr):
    cursor = connStr.cursor()
    print("starting to enter database")    
    for index,row in d.iterrows():
        cursor.execute("INSERT INTO dbo.ComparisionClv([CustomerID],[201110],[201111],[201112],[PredictedClv],[CLV]) values (?, ?,?,?,?,?)", row['CustomerID'], row['201110'] , row['201111'],row['201112'],row['PredictedClv'],row['CLV']) 
        connStr.commit()
    cursor.close()
    connStr.close()
    
def sqlEntryAss(d,connStr):
    cursor = connStr.cursor()
    print("starting to enter database")    
    for index,row in d.iterrows():
        cursor.execute("INSERT INTO dbo.Association([Item1],[Item2],[Item3],[Lift],[Confidence],[Support]) values (?, ?,?,?,?,?)", row['Item1'], row['Item2'] , row['Item3'],row['Lift'],row['Confidence'],row['Support']) 
        connStr.commit()
    cursor.close()
    #connStr.close()    

def Association(store_data):
    group=store_data.groupby('Inv #')['Item Name'].apply(lambda x: "%s" % ','.join(x))
    gf = group.to_frame().reset_index()
    new=gf
    new=new.drop(new.columns[1], axis=1)
    s=gf['Item Name'].str.split(',', expand=True)
    s.head()
    records = []  
    for i in range(0, 99829):  
        records.append([str(s.values[i,j]) for j in range(0, 336)])
    association_rules = apriori(records, min_support=0.0056, min_confidence=0.2, min_lift=3, min_length=3)  
    association_results = list(association_rules)
    it=list()
    conf=list()
    support=list()
    lift=list()
    row=list()
    
    
    for item in association_results:
        global row2
        # first index of the inner list
        # Contains base item and add item
        pair = item[0] 
        print(pair)
        items = [x for x in pair]
        if(len(pair)==2):
            #print("Rule: " + items[0] + " -> " + items[1])
            s=items[0]+','+items[1]+','+'None'
            it.append(s)
            print("i am haere")
            print(s)
        else:
            print("Rule: " + items[0] + " -> " + items[1]+"->"+items[2])
            s=items[0]+','+items[1]+','+items[2]
            it.append(s)
        
        
            
        
        #print("Rule: " + items[0] + " -> " + items[1])
        print(len(pair))
        #s=items[0]+','+items[1]
        
        #it.append(s)
        
        #second index of the inner list
        
        
        print("Support: " + str(item[1]))
        c=str(item[1])
        support.append(c)
        #print(support)
    
        #third index of the list located at 0th
        #of the third index of the inner list
    
        print("Confidence: " + str(item[2][0][2]))
        cnf=str(item[2][0][2])
        conf.append(cnf)
        
        #print("Lift: " + str(item[2][0][3]))
        print("=====================================")
        lt=str(item[2][0][3])
        lift.append(lt)
        
        row2.append(s+','+lt+','+c+','+cnf)
        
    
    for i in row2:
        global b2
        i=i.strip().split(',')
        b2.append(i)
        print(b2)
    b2=pd.DataFrame(b2)
    b2.columns=['Item1','Item2', 'Item3','Lift','Confidence','Support']        
        #asd
    b2['Lift']=b2.Lift.astype(float)
    b2['Confidence']=b2.Confidence.astype(float)
    b2['Support']=b2.Support.astype(float)
    return b2

def CustSeg(df):
    
    data=df
    data.dtypes
    df['Total_Price']=df['Quantity']*df['UnitPrice']

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    y=df['InvoiceDate'].dt.year
    m=df['InvoiceDate'].dt.month
    m=m.astype(str).str.zfill(2)
    y=y.astype(str)
    df['date']=y+m
    df.head()
    df.date = pd.to_numeric(df.date, errors='coerce')
    data=df
    Cust_country=data[['Country','CustomerID']].drop_duplicates()
    Cust_country.head()
    Cust_country_count=Cust_country.groupby(['Country'])['CustomerID'].aggregate('count')
    Cust_country_count=Cust_country_count.reset_index()
    Cust_country_count=Cust_country_count.sort_values('CustomerID', ascending=False)
    Cust_country_count.head()
    country=list(Cust_country_count['Country'])
    Cust_id=list(Cust_country_count['CustomerID'])
    Cust_freq=data[['Country','InvoiceNo','CustomerID']].drop_duplicates()    
    Cust_freq_count = Cust_freq.groupby(["Country","CustomerID"])["InvoiceNo"].aggregate("count").reset_index()
    Cust_freq.head()
    Cust_freq_count.head()
    Cust_freq_count_UK=Cust_freq_count[Cust_freq_count['Country']=="United Kingdom"]
    unique_invoice=Cust_freq_count_UK[['InvoiceNo']].drop_duplicates()
    unique_invoice['Frequency_Band'] = pd.qcut(unique_invoice['InvoiceNo'], 5)
    unique_invoice=unique_invoice[['Frequency_Band']].drop_duplicates()
    unique_invoice
    def f(row):
        if row['InvoiceNo'] <= 13:
            val = 1
        elif row['InvoiceNo'] > 13 and row['InvoiceNo'] <= 25:
            val = 2
        elif row['InvoiceNo'] > 25 and row['InvoiceNo'] <= 38:
            val = 3
        elif row['InvoiceNo'] > 38 and row['InvoiceNo'] <= 55:
            val = 4
        else:
            val = 5
        return val
    Cust_freq_count_UK['Freq_Flag'] = Cust_freq_count_UK.apply(f, axis=1)
    Cust_monetory = data.groupby(["Country","CustomerID"])["Total_Price"].aggregate("sum").reset_index().sort_values(by='Total_Price', ascending=False)
    Cust_monetory_UK=Cust_monetory[Cust_monetory['Country']=="United Kingdom"]    
    unique_price=Cust_monetory_UK[['Total_Price']].drop_duplicates()
    unique_price=unique_price[unique_price['Total_Price'] > 0]
    unique_price['monetory_Band'] = pd.qcut(unique_price['Total_Price'], 5)
    unique_price=unique_price[['monetory_Band']].drop_duplicates()
    unique_price
    def f(row):
        if row['Total_Price'] <= 243:
            val = 1
        elif row['Total_Price'] > 243 and row['Total_Price'] <= 463:
            val = 2
        elif row['Total_Price'] > 463 and row['Total_Price'] <= 892:
            val = 3
        elif row['Total_Price'] > 892 and row['Total_Price'] <= 1932:
            val = 4
        else:
            val = 5
        return val
    Cust_monetory_UK['Monetory_Flag'] = Cust_monetory_UK.apply(f, axis=1)   
    Cust_date_UK=data[data['Country']=="United Kingdom"]
    Cust_date_UK=Cust_date_UK[['CustomerID','date']].drop_duplicates()
    def f(row):
        if row['date'] > 201110:
            val = 5
        elif row['date'] <= 201110 and row['date'] > 201108:
            val = 4;
        elif row['date'] <= 201108 and row['date'] > 201106:
            val = 3
        elif row['date'] <= 201106 and row['date'] > 201104:
            val = 2
        else:
            val = 1
        return val
    Cust_date_UK['Recency_Flag'] = Cust_date_UK.apply(f, axis=1)
    Cust_date_UK = Cust_date_UK.groupby("CustomerID", as_index=False)["Recency_Flag"].max()
    Cust_UK_All=pd.merge(Cust_date_UK,Cust_freq_count_UK[['CustomerID','Freq_Flag']],                     on=['CustomerID'],how='left')
    Cust_UK_All=pd.merge(Cust_UK_All,Cust_monetory_UK[['CustomerID','Monetory_Flag']],                     on=['CustomerID'],how='left')
    Cust_UK_All['Overall_Percentage']=Cust_UK_All['Recency_Flag']+Cust_UK_All['Freq_Flag']+Cust_UK_All['Monetory_Flag']/15*100
    Cust_UK_All.head()
    return Cust_UK_All

def ClvPrepare(af):
        af['Total_Price']=af['Quantity']*af['UnitPrice']
        af['InvoiceDate'] = pd.to_datetime(af['InvoiceDate'])
        y=af['InvoiceDate'].dt.year
        m=af['InvoiceDate'].dt.month
        m=m.astype(str).str.zfill(2)
        y=y.astype(str)
        af['date']=y+m
        mkt=af.groupby(["CustomerID"])["Total_Price"].aggregate("sum").reset_index()
        t=af.groupby(af['date'])['Total_Price'].agg('sum')
        a=['201112','201111','201110','201109','201108','201107']
        new=af
        new = new[new.date.isin(a)]
        pt = new.pivot_table(index='CustomerID', columns='date', values='Total_Price', aggfunc=sum)  
        flattened = pd.DataFrame(pt.to_records())
        pd.DataFrame(mkt, columns=['CustomerID'])
        final=pd.merge(flattened, mkt, on='CustomerID')
        
        cols = [1,2,3]
        final.drop(final.columns[cols],axis=1,inplace=True)
        
        final = final[final['201110'].notnull()]
        final = final[final['201111'].notnull()]
        final = final[final['201112'].notnull()]
        final = final.rename(columns={'Total_Price': 'CLV'})
    
    
        return final
    
def ClvPredTrain(df):
    cleaned_data = df.drop('CustomerID',axis=1)

    cleaned_data .corr()['CLV']
    predictors = cleaned_data.drop('CLV',axis=1)
    targets = cleaned_data.CLV
    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.25,shuffle=False)
    print( 'Predictor — Training :', pred_train.shape, 'Predictor — Testing : ', pred_test.shape)
    model = LinearRegression()
    
    model.fit(pred_train,tar_train)
    print('Coefficients: \n', model.coef_)
    print('Intercept:', model.intercept_)

#Test on testing data

    predictions = model.predict(pred_test)

    predictions

    sklearn.metrics.r2_score(tar_test, predictions)
    
    
    r=int(df.CustomerID.count()*0.25+1)
    predictions=pd.DataFrame(predictions)
    predictions.columns=['PredictedClv']
    pred_test1=pred_test.reset_index()
    result = pd.concat([pred_test1, predictions],axis=1)
    Cid=pd.DataFrame(df.CustomerID.tail(r))
    Cid2=Cid.reset_index()
    result22=result.reset_index()
    PredictionDisplay = pd.merge(Cid2,result,how = 'outer',left_index = True, right_index = True)
    tail=df.tail(r)
    tail=tail.reset_index()
    Comparision = pd.merge(PredictionDisplay,pd.DataFrame(tail['CLV']),how = 'outer',left_index = True, right_index = True)
    PredictionDisplay=PredictionDisplay.drop(PredictionDisplay.columns[[0,2]], axis = 1)
    Comparision=Comparision.drop(Comparision.columns[[0,2]], axis = 1)
    return PredictionDisplay, Comparision, model
        

#! /usr/bin/env python
#
# GUI module generated by PAGE version 4.8.6
# In conjunction with Tcl version 8.6
#    Apr 28, 2019 11:44:01 AM
import sys

try:
    from Tkinter import *
except ImportError:
    from tkinter import *

try:
    import ttk
    py3 = 0
except ImportError:
    import tkinter.ttk as ttk
    py3 = 1

import unknown_support

  
def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root,pred
    root = Tk()
    top = New_Toplevel_1 (root)
    unknown_support.init(root, top)
    root.mainloop()

w = None
def create_New_Toplevel_1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt,pred
    rt = root
    w = Toplevel (root)
    top = New_Toplevel_1 (w)
    unknown_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_New_Toplevel_1():
    global w
    w.destroy()
    w = None


class New_Toplevel_1:   
    global pred,r,model,fitModel,df
    def  __init__(self, top=None):
        global model
        df=pd.read_csv(r"C:\Users\shawa\Documents\OnlineRetail.csv",error_bad_lines=False, encoding = 'unicode_escape')
        store_data=pd.read_csv(r"C:\Users\shawa\Documents\finalHomeplus3 amna.csv")
        gff=pd.read_csv(r"C:\Users\shawa\Documents\OnlineRetail.csv",error_bad_lines=False, encoding = 'unicode_escape')
        pf=Association(store_data)
        sqlEntryAss(pf,connStr)
        
        myCust=CustSeg(gff)
        sqlEntryRfm(myCust,connStr)
        
        dp=ClvPrepare(df) 
        result=ClvPredTrain(dp)
        model=result[2]
        sqlEntryClv(result[1],connStr)
        
        
        #df=ClvPrepare(af)
#        df.head()
#        cols = [0,2,3,4]
#        df.drop(df.columns[cols],axis=1,inplace=True)
#        df.head()
#        df = df[df['201110'].notnull()]
#        df = df[df['201111'].notnull()]
#        df = df[df['201112'].notnull()]
#        print("displaying head")
#        df.head()
#        df = df.rename(columns={'Total_Price': 'CLV'})
#        cleaned_data = df.drop('CustomerID',axis=1)
#        cleaned_data .corr()['CLV']
#        predictors = cleaned_data.drop('CLV',axis=1)
#        targets = cleaned_data.CLV
#        pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.25,shuffle=False)
#        print( 'Predictor — Training :', pred_train.shape, 'Predictor — Testing : ', pred_test.shape)
#        model = LinearRegression()
#        model.fit(pred_train,tar_train)
#        print('Coefficients: \n', model.coef_)
#        print('Intercept:', model.intercept_)
#        predictions = model.predict(pred_test)
#        predictions
#        z=sklearn.metrics.r2_score(tar_test, predictions)*100
#        print("here is the score",z)
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        self._bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        self._fgcolor = '#000000'  # X11 color: 'black'
        self._compcolor = '#d9d9d9' # X11 color: 'gray85'
        self._ana1color = '#d9d9d9' # X11 color: 'gray85' 
        self._ana2color = '#ececec' # Closest X11 color: 'gray92' 
        self.font3 = "-family {Segoe UI} -size 18 -weight bold -slant "  \
            "roman -underline 0 -overstrike 0"
        self.font4 = "-family {Segoe UI} -size 14 -weight bold -slant "  \
            "roman -underline 0 -overstrike 0"
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=self._bgcolor)
        self.style.configure('.',foreground=self._fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', self._compcolor), ('active',self._ana2color)])

        top.geometry("600x450+640+225")
        top.title("New Toplevel 1")
        top.configure(background="#d9d9d9")



        self.Label1 = Label(top)
        self.Label1.place(relx=0.23, rely=0.07, height=43, width=326)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font=self.font3)
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Customer CLV Prediction''')

        self.Entry1 = Entry(top)
        self.Entry1.place(relx=0.15, rely=0.22, relheight=0.07, relwidth=0.07)
        self.Entry1.configure(background="white")
        self.Entry1.configure(disabledforeground="#a3a3a3")
        self.Entry1.configure(font="TkFixedFont")
        self.Entry1.configure(foreground="#000000")
        self.Entry1.configure(insertbackground="black")
        self.Entry1.configure(width=44)

        self.Entry2 = Entry(top)
        self.Entry2.place(relx=0.35, rely=0.22, relheight=0.07, relwidth=0.07)
        self.Entry2.configure(background="white")
        self.Entry2.configure(disabledforeground="#a3a3a3")
        self.Entry2.configure(font="TkFixedFont")
        self.Entry2.configure(foreground="#000000")
        self.Entry2.configure(highlightbackground="#d9d9d9")
        self.Entry2.configure(highlightcolor="black")
        self.Entry2.configure(insertbackground="black")
        self.Entry2.configure(selectbackground="#c4c4c4")
        self.Entry2.configure(selectforeground="black")

        self.Entry3 = Entry(top)
        self.Entry3.place(relx=0.25, rely=0.22, relheight=0.07, relwidth=0.07)
        self.Entry3.configure(background="white")
        self.Entry3.configure(disabledforeground="#a3a3a3")
        self.Entry3.configure(font="TkFixedFont")
        self.Entry3.configure(foreground="#000000")
        self.Entry3.configure(highlightbackground="#d9d9d9")
        self.Entry3.configure(highlightcolor="black")
        self.Entry3.configure(insertbackground="black")
        self.Entry3.configure(selectbackground="#c4c4c4")
        self.Entry3.configure(selectforeground="black")


        

        self.Label2 = Label(top)
        self.Label2.place(relx=0.25, rely=0.42, height=43, width=326)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font=self.font3)
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Prediction answer''')

        self.Button1 = Button(top)
        self.Button1.place(relx=0.42, rely=0.73, height=51, width=89)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(font=self.font4)
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''predict''')
        self.Button1.configure(command=self.predict)
        
        
        
        self.Text1 = Text(top)
        self.Text1.place(relx=0.27, rely=0.6, relheight=0.07, relwidth=0.47)
        self.Text1.configure(background="white")
        self.Text1.configure(font="TkTextFont")
        self.Text1.configure(foreground="black")
        self.Text1.configure(highlightbackground="#d9d9d9")
        self.Text1.configure(highlightcolor="black")
        self.Text1.configure(insertbackground="black")
        self.Text1.configure(selectbackground="#c4c4c4")
        self.Text1.configure(selectforeground="black")
        self.Text1.configure(width=284)
        self.Text1.configure(wrap=WORD)
        
    
        
    
        
    def predict(self):
        
        in1=int(self.Entry1.get())
        in2=int(self.Entry2.get())
        in3=int(self.Entry3.get())
        arr=np.array([in1,in2,in3]).reshape(1,-1)
        print("my array:",arr)
        pred=("£",fitModel(arr,model))
        #pred=("£",fitModel(arr,model))
        self.Text1.insert(0.0,pred)
        
    def fitModel(arr,model):
        new_data = arr
        new_data
        new_pred=model.predict(arr)
        print('The CLV for the new customer is : $',new_pred[0])
        pred=new_pred[0]
        return pred
    
    
        
        
        
if __name__ == '__main__':
    vp_start_gui()


        