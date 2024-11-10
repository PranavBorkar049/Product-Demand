import flask
from flask import Flask, render_template, request, redirect
from flask import Flask, render_template, request, redirect, url_for,session, send_from_directory
# from eval import main
import re
import os
import pandas as pd








from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded files

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email=db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    images = db.relationship('Images', backref='user', lazy=True)
    # def __repr__(self):
    #     return f'<User {self.username}>'

class Images(db.Model):
    image_id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    filename = db.Column(db.String(50), nullable=False)
    data = db.Column(db.LargeBinary)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # def __repr__(self):
    #     return f'<Image {self.filename}>'



@app.before_request
def create_tables():
    db.create_all()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'confirm_password' in request.form:
        username = request.form['username']
        password = request.form['password']
        email=request.form['email']
        confirm_password = request.form['confirm_password']
        account = User.query.filter_by(username=username).first()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must not contain any special characters!'
        elif not username or not password or not confirm_password:
            msg = 'Please fill out the form !'
        elif password != confirm_password:
            msg = 'Passwords do not match.'
        else:
            new_user = User(username=username, password=password,email=email)
            db.session.add(new_user)
            db.session.commit()
            msg = 'You have successfully registered!'
            # return render_template('signup.html', msg=msg)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
        return redirect(url_for('login'))
    return render_template('signup.html',msg=msg)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        session['logged_in'] = True
        session['email'] = email
        global user
        user = User.query.filter_by(email=email).first()
        session['name'] = user.username
        if not user or user.password != password:
            error = 'Invalid username or password.'
            return render_template('login.html', error=error)
        #return redirect(url_for('upload'))
        return render_template('index.html',user=session['name'])
    return render_template('login.html')






#Basic Web Pages
#-------------------------------------------------------------------------------------------
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/after")
def index():
    return render_template("index.html")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/model")
def model():
    return render_template("model.html")



@app.route('/upload', methods=['POST'])
def upload():
    csv_file = request.files.get('csv_file')
    
    
    if not csv_file:
        return "No file part"

    if csv_file.filename == '':
        return "No selected file"

    if csv_file and csv_file.filename.endswith('.csv'):
        csv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
        csv_file.save(csv_filepath)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_filepath)
        df=df.iloc[:21,:]

        # Convert DataFrame to HTML table
        data_html = df.to_html(classes='table table-striped', index=False)
        

        return render_template('all.html', data=data_html)
    
    return render_template("index.html")

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import Flask, render_template
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

@app.route('/results-randomforest')
def random_forest_results():
    # Load and preprocess data (simplified for example purposes)
    df = pd.read_csv('uploads/data.csv')
    
    # Feature creation
    df['key'] = df['week'].astype(str) + '_' + df['store_id'].astype(str)
    df['day_1'] = df['units_sold'].shift(-1)
    df['day_2'] = df['units_sold'].shift(-2)
    df['day_3'] = df['units_sold'].shift(-3)
    df['day_4'] = df['units_sold'].shift(-4)
    df = df.dropna()

    # Prepare features and target
    x1, x2, x3, x4, y = df['day_1'], df['day_2'], df['day_3'], df['day_4'], df['units_sold']
    x = np.concatenate((x1.values.reshape(-1, 1), 
                        x2.values.reshape(-1, 1), 
                        x3.values.reshape(-1, 1), 
                        x4.values.reshape(-1, 1)), axis=1)
    y = y.values.reshape(-1, 1)

    # Train/test split
    split_percentage = 15
    test_split = int(len(df) * (split_percentage / 100))
    X_train, X_test, y_train, y_test = x[:-test_split], x[-test_split:], y[:-test_split], y[-test_split:]

    # RandomForest model
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train, y_train.ravel())

    # R-squared score
    r_sq = rf_regressor.score(X_test, y_test)

    # Predictions
    y_pred = rf_regressor.predict(X_test)

    # Plot: Actual vs Predicted Sales
    plt.figure(figsize=(12, 8))
    plt.plot(y_pred[-100:], label='Predictions')
    plt.plot(y_test[-100:], label='Actual Sales')
    plt.legend(loc="upper left")
    plt.title("Actual vs Predicted Sales - Random Forest")
    
    # Convert plot to PNG image and then to base64 string
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('results.html', r_sq=r_sq, plot_url=plot_url)


import xgboost
@app.route('/results-xgboost')
def xgboost_results():
    df = pd.read_csv('uploads/data.csv')
    
    df['key'] = df['week'].astype(str) + '_' + df['store_id'].astype(str)
    df['day_1'] = df['units_sold'].shift(-1)
    df['day_2'] = df['units_sold'].shift(-2)
    df['day_3'] = df['units_sold'].shift(-3)
    df['day_4'] = df['units_sold'].shift(-4)
    df = df.dropna()

    x1, x2, x3, x4, y = df['day_1'], df['day_2'], df['day_3'], df['day_4'], df['units_sold']
    x = np.concatenate((x1.values.reshape(-1, 1), 
                        x2.values.reshape(-1, 1), 
                        x3.values.reshape(-1, 1), 
                        x4.values.reshape(-1, 1)), axis=1)
    y = y.values.reshape(-1, 1)

    split_percentage = 15
    test_split = int(len(df) * (split_percentage / 100))
    X_train, X_test, y_train, y_test = x[:-test_split], x[-test_split:], y[:-test_split], y[-test_split:]

    xgb_regressor = xgboost.XGBRegressor()
    xgb_regressor.fit(X_train, y_train.ravel())
    r_sq_xgb = xgb_regressor.score(X_test, y_test)

    y_pred_xgb = xgb_regressor.predict(X_test)

    plt.figure(figsize=(12, 8))
    plt.plot(y_pred_xgb[-100:], label='Predictions')
    plt.plot(y_test[-100:], label='Actual Sales')
    plt.legend(loc="upper left")
    plt.title("Actual vs Predicted Sales - XGBoost")
    
    img_xgb = BytesIO()
    plt.savefig(img_xgb, format='png')
    img_xgb.seek(0)
    plot_url_xgb = base64.b64encode(img_xgb.getvalue()).decode()

    return render_template('result-xg.html', r_sq=r_sq_xgb, plot_url=plot_url_xgb)

from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.impute import SimpleImputer

@app.route('/features')
def features():
    # Load the dataset
    df = pd.read_csv('uploads/data.csv')

    # Impute missing values in numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    # Create a new 'key' column by combining 'week' and 'store_id'
    df['key'] = df['week'].astype(str) + '_' + df['store_id'].astype(str)

    # Drop unnecessary columns
    df = df.drop(['record_ID', 'week', 'store_id', 'sku_id', 'total_price', 
                  'base_price', 'is_featured_sku', 'is_display_sku'], axis=1)

    # Group by the new 'key' and sum the remaining columns
    grouped_df = df.groupby('key').sum()

    # Create the plot
    plt.figure(figsize=(12, 8))
    grouped_df[:100].plot()
    plt.title("Sample Data Visualization")
    
    # Save the plot to a BytesIO object and encode it
    img_features = BytesIO()
    plt.savefig(img_features, format='png')
    img_features.seek(0)
    plot_url = base64.b64encode(img_features.getvalue()).decode()
    plt.close()  # Close the plot to free up memory

    # Render the HTML template and pass the base64 encoded plot image
    return render_template('features.html', plot_url=plot_url)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('email', None)
    return redirect(url_for('home'))   
 

# Commands to run
#-------------------------------------------------------------------------------------------
# export FLASK_APP=server.py
# export FLASK_DEBUG=1
# python -m flask run --host 0.0.0.0 --port 5000

if __name__ == '__main__':
    app.run(debug=True)