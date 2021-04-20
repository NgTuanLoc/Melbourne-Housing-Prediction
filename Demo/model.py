import pickle
import sklearn
import pandas as pd
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv("model/Datasets/Melbourne_housing_FULL.csv")


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


num_attrb_selected = ["Rooms", "Distance", "Bedroom2", "Bathroom", "Car", "Landsize", "Lattitude", "Longtitude"]

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(num_attrb_selected)),    
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

cat_attrb_selected = ["Suburb", "Type", "Method", "SellerG", "Date", "CouncilArea",   "Regionname"]

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(cat_attrb_selected)),
        ("imputer", MostFrequentImputer()),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))    
    ])



full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attrb_selected),
        ("cat", cat_pipeline, cat_attrb_selected),
    ])

full_pipeline.fit(data[data.Price.notnull()])



SVM = pickle.load(open('model/Pickle/SVM.pkl', 'rb'))
KNN = pickle.load(open('model/Pickle/KNN.pkl', 'rb'))
RF = pickle.load(open('model/Pickle/RF.pkl', 'rb'))
SVM_Grid = pickle.load(open('model/Pickle/svm_grid.pkl','rb'))
KNN_Grid = pickle.load(open('model/Pickle/knn_grid.pkl','rb'))
RF_Random = pickle.load(open('model/Pickle/rf_random.pkl','rb'))
# print(SVM.predict(full_pipeline.transform(data.head(1))))

test = [2, 2.5, 2, 1, 1, 202, -37.7996, 144.9984, "Abbotsford", "h", "S", "Biggin", "3/12/2016", "Yarra City Council", "Northern Metropolitan"]

features = num_attrb_selected + cat_attrb_selected

def convert(prediction):
    if prediction==1:
        return "Low"
    elif prediction==2:
        return "Medium"
    else:
        return "High"
