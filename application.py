from flask import Flask,render_template,request,jsonify
import pickle
import numpy
import pandas

app = Flask(__name__,template_folder="templates")

MODEL_PATH = "artifacts/models/random_forrest_model.pkl"

with open(MODEL_PATH,'rb') as model_file:
    model = pickle.load(model_file)

FEATURE_NAMES = ['Age', 'Fare', 'Pclass', 'Sex', 'Embarked', 'Familysize', 'Isalone',
       'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']