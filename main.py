import pandas as pd
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.utils import secure_filename
import os
from sklearn.model_selection import train_test_split
import shutil
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from oversampling import X,y,X_test,y_test
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import pygal

app=Flask(__name__)
app.config['UPLOAD_FOLDER']=r"E:\project\SOURCE_CODE\uploadcsv"
app.config['SECRET_KEY']='b0b4fbefdc48be27a6123605f02b6b86'
from sklearn.metrics import confusion_matrix
@app.route("/")
def main():
    return render_template("index.html")

@app.route("/upload",methods=['POST','GET'])
def upload():
    if request.method=='POST':
        myfile = request.files['filename']
        ext = os.path.splitext(myfile.filename)[1]
        print("1111!!!!!!")
        print(ext)
        if ext.lower() == ".csv":
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.mkdir(app.config['UPLOAD_FOLDER'])
            myfile.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(myfile.filename)))
            flash('The data is loaded successfully', 'success')
            return render_template('upload.html')
        else:
            flash('Please upload a CSV type document only', 'warning')
            return render_template('upload.html')
    return render_template("upload.html")

@app.route('/view')
def view():
    #dataset
    myfile=os.listdir(app.config['UPLOAD_FOLDER'])
    global full_data
    full_data=pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"],myfile[0]))
    return render_template('view_dataset.html', col=full_data.columns.values, df=list(full_data.values.tolist()))

@app.route('/split', methods=['POST','GET'])
def split():
    if request.method=="POST":
        test_size=float(request.form['size'])
        global X_train_SMOTE, y_train_SMOTE
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        SMOT = SMOTE()
        X_train_SMOTE, y_train_SMOTE = SMOT.fit_resample(X_train, y_train)
        flash('The dataset is transformed and split successfully','success')
        return redirect(url_for('model_performance'))
    return render_template('split_dataset.html')

@app.route('/model_performance', methods=['GET','POST'])
def model_performance():
    if request.method=="POST":
        model_no=int(request.form['algo'])
        if model_no==0:
            print("U have not selected any model")
        elif model_no==1:
            model = SVC()
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            cm1=confusion_matrix(y_test,y_pred)
            print(cm1)
            total1=sum(sum(cm1))
            sensitivity=(cm1[0,0]/cm1[1,1])+total1
            print(sensitivity)
            import numpy as np
            print(np.unique(y_pred))
            global re1,pr1,accuracyscore3
            re1 = recall_score(y_test, y_pred, average='micro')
            pr1 = precision_score(y_test, y_pred, average='micro')
            roc1 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore3 = accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore3, model=model_no)
        elif model_no == 2:
            model = RandomForestClassifier(n_estimators=20)
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            cm1 = confusion_matrix(y_test, y_pred)
            total1 = sum(sum(cm1))
            sensitivity2 = (cm1[0, 0] / cm1[1, 1]) + total1
            print(sensitivity2)
            global re2,pr2,accuracyscore1
            re2 = recall_score(y_test, y_pred, average='micro')
            pr2 = precision_score(y_test, y_pred, average='micro')
            roc2 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore1 = accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore1, model=model_no)

        elif model_no == 3:
            model = DecisionTreeClassifier(criterion='entropy', random_state=0)
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            cm1 = confusion_matrix(y_test, y_pred)
            total2 = sum(sum(cm1))
            sensitivity3 = (cm1[0, 0] / cm1[1, 1]) + total2
            print(sensitivity3)
            re3 = recall_score(y_test, y_pred, average='micro')
            pr3 = precision_score(y_test, y_pred, average='micro')
            roc3 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore, model=model_no)
        elif model_no == 4:
            model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            cm1 = confusion_matrix(y_test, y_pred)
            total1 = sum(sum(cm1))
            sensitivity4 = (cm1[0, 0] / cm1[1, 1]) + total1
            print(sensitivity4)
            re4 = recall_score(y_test, y_pred, average='micro')
            pr4 = precision_score(y_test, y_pred, average='micro')
            roc4 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore, model=model_no)
        elif model_no == 5:
            model = XGBClassifier()
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
            cm1 = confusion_matrix(y_test, y_pred)
            total1 = sum(sum(cm1))
            sensitivity5 = (cm1[0, 0] / cm1[1, 1]) + total1
            print("sensitivity score",sensitivity5)
            global re5,pr5,accuracyscore2
            re5 = recall_score(y_test, y_pred, average='micro')
            pr5 = precision_score(y_test, y_pred, average='micro')
            roc5 = roc_auc_score(y_true=y_test, y_score=y_pred, average='micro')
            accuracyscore2= accuracy_score(y_test, y_pred)
            return render_template('train_model.html', acc=accuracyscore2, model=model_no)
    return render_template('train_model.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=='POST':
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = request.form['f9']
        f10 = request.form['f10']
        print("11111111")
        all_obj_vals=[[float(f1),float(f2),float(f3),float(f4),float(f5),float(f6),float(f7),float(f8),float(f9),float(f10)]]
        model = RandomForestClassifier(n_estimators=6)
        model.fit(X_train_SMOTE, y_train_SMOTE)
        pred=model.predict(all_obj_vals)
        return render_template('predict.html',pred=pred)
    return render_template('predict.html')

@app.route("/bar_chart")
def bar_chart():
    line_chart = pygal.Bar(width=700,height=300)
    line_chart.title = 'Classification of stroke disease using machine learning algorithms'
    line_chart.add('PRECISION SCORE',[pr2])
    line_chart.add('RECALL SCORE',[re2])
    line_chart.add('ACCURACY SCORE',[accuracyscore1])
    line_chart.add('PRECISION SCORE', [pr5])
    line_chart.add('RECALL SCORE', [re5])
    line_chart.add('ACCURACY SCORE', [accuracyscore2])
    #line_chart.add('PRECISION SCORE', [pr1])
    #line_chart.add('RECALL SCORE', [re1])
    #line_chart.add('ACCURACY SCORE', [accuracyscore3])
    graph_data = line_chart.render()
    return render_template('bar_chart.html', graph_data=graph_data)

if(__name__)==("__main__"):
    app.run(debug=True)


