from time import time
import streamlit as st

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LGR
# from sklearn.decomposition import PCA

from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Analytical performace of traditional feature selection methods on high dimensional data")
st.write("""
## Try different algorithms and their selection methods 
""")

dataset_name = st.sidebar.selectbox(
    "Select Dataset", ("Breast Cancer", "Vehicle loan", "Loan dataset"))
st.write("Dataset chosen :", dataset_name)

direction_name = st.sidebar.selectbox(
    "Select algorithm", ("Forward selection", "Backward selection"))
st.write("Algorithm chosen", direction_name)

float_selected = st.sidebar.selectbox(
    "Floating", ("True", "False"))

target_column_name = st.sidebar.text_input('Target')

drop_column_strings = st.sidebar.text_input(
    'Drop column names\nEnter with \"comma(,)" seperated')


drop_column_array = []
if drop_column_strings != "":
    drop_column_array = drop_column_strings.split(",")
else:
    st.info("Enter the columns to be dropped")

drop_column_array = np.array(drop_column_array)

if target_column_name == "":
    st.warning("Enter the target")


def get_dataset(dataset_name):
    if dataset_name == "Vehicle loan":
        data = pd.read_csv(
            "https://raw.githubusercontent.com/dakshina13/dimensionality-reduction/master/vehicle-loan/dataset/train.csv")
    elif dataset_name == "Breast Cancer":
        data = pd.read_csv(
            "https://raw.githubusercontent.com/dakshina13/dimensionality-reduction/master/cancer/data3.csv")
    elif dataset_name == "Loan dataset":
        data = pd.read_csv(
            "https://raw.githubusercontent.com/atulpatelDS/Data_Files/master/Loan_Dataset/loan_data_set.csv")
    return data


def get_direction(direction_name):
    if direction_name == "Forward selection":
        return True
    else:
        return False


def get_float_value(float_value):
    if float_value == "True":
        return True
    else:
        return False


directionValue = get_direction(direction_name)

raw_data = get_dataset(dataset_name)

float_value = get_float_value(float_selected)

#st.write("Initial shape of dataset", raw_data.shape)


if "Unnamed: 32" in raw_data.columns.array:
    raw_data = raw_data.drop("Unnamed: 32", axis=1)
# st.write("shape of dataset", raw_data.shape)

raw_data = raw_data.dropna()

st.write("Initial shape of dataset", raw_data.shape)


st.write("All columns\n", raw_data.columns.array)

# st.write("Drop column", drop_column_array.shape)
# st.write("Drop column ", drop_column_array)


def drop_selected_columns(data, drop_column_array):
    for col in drop_column_array:
        data = data.drop(col, axis=1)
    return data


def plot_graph(sfs1, graphName):
    fig = plt.subplots()
    plot_sfs(sfs1.get_metric_dict(confidence_interval=0.95), kind='std_err')
    plt.title(graphName)
    plt.grid()
    st.pyplot()


def encode_label(train):
    label_encoder = preprocessing.LabelEncoder()
    for k, v in train.dtypes.items():
        if v not in ["int32", "int64", "float64"]:
            # print("For k "+k)
            # print(train[k].unique())
            train[k] = label_encoder.fit_transform(train[k])
            # print(train[k].unique())
    return train

def predictAccuracy(x,y,columns,shape,label,titleLabel):
    lgr=LGR(max_iter=10000)
    new_x=x[columns]
    x_train,x_test,y_train,y_test=train_test_split(new_x,y,test_size=0.3)
    train_time_start=time()
    lgr.fit(x_train,y_train)
    st.subheader("Training using Logistic regression Model "+titleLabel)
    st.write("Training with ",shape,label)
    st.write("Training time ", time()-train_time_start," seconds")
    acc=lgr.score(x_test,y_test)
    st.write("Accuracy ",acc)



def select_features(X, y, forwardValue, floatValue):
    feature_names = tuple(X.columns)
    start_time = time()
    sfs1 = SFS(  # knn(n_neighbors=3),
        # rfc(n_jobs=8),
        LGR(max_iter=10000),
        k_features='best',
        forward=forwardValue,
        floating=floatValue,
        verbose=2,
        # scoring = 'neg_mean_squared_error',  # sklearn regressors
        scoring='accuracy',  # sklearn classifiers
        cv=0)
    sfs1 = sfs1.fit(X, y, custom_feature_names=feature_names)
    st.write("Selected features")
    st.write(sfs1.k_feature_names_)
    features_list = np.array(sfs1.k_feature_names_)
    st.write("Selected features shape ", features_list.shape)
    st.write("Time taken ", (time() - start_time), " seconds")
    graphName = ""
    if forwardValue == True and floatValue == True:
        graphName = "Sequential Forward Floating Selection"
    elif forwardValue == False and floatValue == True:
        graphName = "Sequential Backward Floating selection"
    elif forwardValue == True and floatValue == False:
        graphName = "Sequential Forward Selection"
    else:
        graphName = "Sequential Backward selection"
    st.write("Accuracy ",sfs1.k_score_)
    plot_graph(sfs1, graphName)
    predictAccuracy(X,y,X.columns,X.shape[1]," features or with all the features","without feature selection")
    predictAccuracy(X,y,features_list,features_list.shape[0]," features selected","with feature selection")



def dataset_ready(X, y, forwardValue, floatValue):
    st.write("Ready to select features")
    st.write("Shape of dataset before feature selection", X.shape)
    st.write("Number of classes in output or target", len(np.unique(y)))
    select_features(X, y, forwardValue, floatValue)
    # plot_graph()


buttonPressed = st.sidebar.button("Run")

if buttonPressed:
    buttonPressed = False
    if target_column_name != "":
        encoded_data = encode_label(raw_data)
        target = encoded_data[target_column_name]
        encoded_data = encoded_data.drop(target_column_name, axis=1)
        encoded_data = drop_selected_columns(encoded_data, drop_column_array)
        dataset_ready(encoded_data, target, directionValue, float_value)
