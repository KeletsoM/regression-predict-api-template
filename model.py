"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """

    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Pickup Lat','Pickup Long',
                                        #'Destination Lat','Destination Long']]
    # ------------------------------------------------------------------------
    Train = pd.read_csv('utils/data/train_data.csv')
    #Riders = pd.read_csv('utils/data/riders.csv')

    train2= Train.drop(['Vehicle Type', 'Personal or Business','Placement - Day of Month',
                    'Placement - Weekday (Mo = 1)','Confirmation - Day of Month','Confirmation - Weekday (Mo = 1)',
                   'Temperature', 'Precipitation in millimeters','Arrival at Pickup - Day of Month',
                    'Arrival at Pickup - Weekday (Mo = 1)','Pickup - Day of Month','Pickup - Weekday (Mo = 1)',
                    'Arrival at Destination - Day of Month','Arrival at Destination - Weekday (Mo = 1)','User Id'], axis=1)

    train3 = train2.drop(['Placement - Time','Confirmation - Time',
                      'Arrival at Pickup - Time','Pickup - Time',
                      'Arrival at Destination - Time','Pickup Lat','Pickup Long','Destination Lat','Destination Long',
                      'Order No'], axis=1)
    
    feature_vector_df1 = feature_vector_df.drop(['Vehicle Type', 'Personal or Business','Placement - Day of Month',
                    'Placement - Weekday (Mo = 1)','Confirmation - Day of Month','Confirmation - Weekday (Mo = 1)',
                   'Temperature', 'Precipitation in millimeters','Arrival at Pickup - Day of Month',
                    'Arrival at Pickup - Weekday (Mo = 1)','Pickup - Day of Month','Pickup - Weekday (Mo = 1)','User Id'], axis=1)

    feature_vector_df2 = feature_vector_df1.drop(['Placement - Time','Confirmation - Time',
                      'Arrival at Pickup - Time','Pickup - Time','Pickup Lat','Pickup Long','Destination Lat','Destination Long',
                      'Order No'], axis=1)

    feature_vector_df2.drop(['No_Of_Orders','Age','Average_Rating','No_of_Ratings'], axis=1 , inplace=True)

    train3['train'] = 1
    feature_vector_df2['train'] = 0

    combined = pd.concat([train3,feature_vector_df2])
    combined = combined[['Platform Type', 'Distance (KM)', 'Rider Id','train','Time from Pickup to Arrival']]
    combined.columns = [col.replace(' ','_') for col in combined.columns]
    combined.columns = [col.replace('(','') for col in combined.columns]
    combined.columns = [col.replace(')','') for col in combined.columns]

    df_dummies = pd.get_dummies(combined,columns=['Platform_Type','Rider_Id'], drop_first=True)

    Test2 = df_dummies[df_dummies['train']== 0]
    Test2.drop(['Time_from_Pickup_to_Arrival'],axis=1,inplace=True)
    Test2.drop(['train'],axis=1,inplace=True)

    
    return Test2

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    print(prediction[0].tolist())
    return prediction[0].tolist()
