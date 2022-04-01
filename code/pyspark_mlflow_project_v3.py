# Databricks notebook source

#import requird packages
from pyspark.sql.functions import *
import urllib.request
import os
import numpy as np
from pyspark.sql.types import * 
from pyspark.sql.functions import col, lit
from pyspark.sql.functions import udf
from datetime import datetime
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
from pyspark.sql.types import DoubleType, IntegerType, DateType
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
import matplotlib
import matplotlib.pyplot as plt
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
import pandas as pd


#Read data
def read_data(demographicsFile, productsFile, transactionsFile):
    try:
        demographicsDF = spark.read.csv(demographicsFile, header="true", inferSchema="true")
        productsDF = spark.read.csv(productsFile, header="true", inferSchema="true")
        transactionsDF = spark.read.csv(transactionsFile, header="true", inferSchema="true")

        return transactionsDF
    except Exception as e:
        print(e)

#prepare data for recommendation system
def prepareData(dataset, seed = 8451, fraction=0.1, bins = [0,2,6,31,float('Inf')], basket=True):
    try:
        #sample subset from the entire data
        #dataset = dataset.sample(fraction=fraction, seed=seed)
        
        #change data type of columns
        dataset = dataset.withColumn('product_id', col('product_id').cast(IntegerType()))
        dataset = dataset.withColumn('household_id', col('household_id').cast(IntegerType()))
        
        if(basket):
            preppedDataDF = dataset.groupBy('household_id', 'product_id').agg(count("basket_id").alias("count"))
            #discretizer = QuantileDiscretizer(numBuckets=5, relativeError=0.01, inputCol="count", outputCol="rating")
            #preppedDataDF = discretizer.fit(preppedDataDF).transform(preppedDataDF)
            bucketizer = Bucketizer(splits = bins, inputCol="count", outputCol="rating")
            preppedDataDF = bucketizer.setHandleInvalid("keep").transform(preppedDataDF)
            preppedDataDF = preppedDataDF.withColumn("rating", preppedDataDF.rating+1)
        else:
            preppedDataDF = dataset.groupBy('household_id', 'product_id').agg(sum("quantity").alias("total"))
            #discretizer = QuantileDiscretizer(numBuckets=5, relativeError=0.01, inputCol="total", outputCol="rating")
            #preppedDataDF = discretizer.fit(preppedDataDF).transform(preppedDataDF)
            bucketizer = Bucketizer(splits = bins, inputCol="total", outputCol="rating")
            preppedDataDF = bucketizer.setHandleInvalid("keep").transform(preppedDataDF)
            preppedDataDF = preppedDataDF.withColumn("rating", preppedDataDF.rating+1)

        return preppedDataDF
    
    except Exception as e:
        print(e)

#train ALS model and run MLflow experiments
def train_als(trainingData, testData, maxIter, regParam, userCol, itemCol, ratingCol, coldStartStrategy, implicitPrefs, predictionCol, metricName):
    # Evaluate metrics
    def evaluateALS(testData, model, metricName, ratingCol, predictionCol):
        predictions = model.transform(testData)
        evaluator = RegressionEvaluator(metricName=metricName, labelCol=ratingCol, predictionCol=predictionCol)
        rmse = evaluator.evaluate(predictions)
        return rmse

    # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
    with mlflow.start_run():
        
        als = ALS(maxIter=maxIter, regParam=regParam, userCol=userCol, itemCol=itemCol, ratingCol=ratingCol, coldStartStrategy=coldStartStrategy, implicitPrefs=implicitPrefs, seed = 8451)
        alsmodel = als.fit(trainingData)
        rmse = evaluateALS(testData, alsmodel, metricName, ratingCol, predictionCol)
        
        # Print out model metrics
        print("ALE model (regParam=%f, maxIter=%f):" % (regParam, maxIter))
        print("  RMSE: %s" % rmse)

        # Log hyperparameters for mlflow UI
        #mlflow.log_param("coldStartStrategy", coldStartStrategy)
        #mlflow.log_param("implicitPrefs", implicitPrefs)
        mlflow.log_param("reg_param", regParam)
        mlflow.log_param("max_iter", maxIter)
        
        # Log evaluation metrics
        mlflow.log_metric("rmse", rmse)
        
        # Log the model itself
        mlflow.spark.log_model(alsmodel, "model")
        modelpath = "/dbfs/mlflow/ip_project/model-%f-%f" % (regParam, maxIter)
        mlflow.spark.save_model(alsmodel, modelpath)

#function to calulate score and return predictions
def get_score(model_path, test):
    try: 
        logged_model = model_path
        #load mlflow model
        model = mlflow.spark.load_model(logged_model)
        # Make predictions on test documents
        prediction = model.transform(test)

        return prediction

    except exception as e:
        print(e)

#main function
def main():

    demographicsFile = "/FileStore/tables/ipdata/demographics.csv"
    productsFile = "/FileStore/tables/ipdata/products.csv"
    transactionsFile = "/FileStore/tables/ipdata/transactions_full.csv"

    dataset = read_data(demographicsFile, productsFile, transactionsFile)

    #set parameters
    seed = 8451
    fraction = 0.01

    #get prepared data
    preppedDataDF = prepareData(dataset, seed, fraction, [0,2,6,31,float('Inf')], True)
    preppedDataDF.show(5, truncate = True)

    #split dataset into train and test
    (trainingData, testData) = preppedDataDF.randomSplit([0.7, 0.3], seed=8451)

    #the following command removes data from prior runs, allowing you to re-run the notebook later without error.
    dbutils.fs.rm("dbfs:/mlflow/ip_project/", True)

    #define parameters for ALS model
    maxIter = [5,10,20]
    regParam = [0.01,0.005,0.1]
    userCol = "household_id"
    itemCol = "product_id"
    ratingCol = "rating"
    coldStartStrategy = "drop"  
    implicitPrefs = True
    metricName = "rmse"
    predictionCol = "prediction"

    #run Experiments
    for iter in maxIter:
        for reg in regParam:
            train_als(trainingData, testData, iter, reg, userCol, itemCol, ratingCol, coldStartStrategy, implicitPrefs, predictionCol, metricName)

    #initiate mlflow client
    client = MlflowClient()
    print('Loaded MLflow Client')
    #get list of all experiments
    client.list_experiments()

    # Replace experiment_num with the appropriate experiment number based on the list of experiments above.
    experiment_num = 0 # FILL IN!

    experiment_id = client.list_experiments()[experiment_num].experiment_id
    runs_df = mlflow.search_runs(experiment_id)
    display(runs_df)

    #model path
    model_path_model = 'runs:/e9ade5bc77d8441db7b011fdb699864e/model'
    #create test data
    test = spark.createDataFrame([(1, 840361)], ["household_id", "product_id"])

    prediction = get_score(model_path, test)
    prediction.show()

if __name__ == "__main__":

    main()