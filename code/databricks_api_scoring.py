#!/usr/bin/env python
# coding: utf-8

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset, url, key):
    '''This function takes data, url of hosted model and api key as input
       uses model served on databricks to predict recommendations and returns predicted scores as json'''
    try:
        url = url
        headers = {'Authorization': f'Bearer {key}'}
        data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
        response = requests.request(method='POST', headers=headers, url=url, json=data_json)
        if response.status_code != 200:
            raise Exception(f'Request failed with status {response.status_code}, {response.text}')
        return response.json()
    
    except Exception as e:
        print(e)
        
#example input dataframe
data =  pd.DataFrame([{"household_id": 1,"product_id": 840361,}])

#call score_model
rcommendation = score_model(data)
print(rcommendation)