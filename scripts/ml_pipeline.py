# This scripts performs the following tasks: ML Pipeline Creation, Evaluation of Model, and Model Persistence using PySpark.

# Part 2 : Machine Learning Pipeline Creation

# Importing required libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
import pandas as pd
import numpy as np
import os

# Task 1 : Load data from "mpg-cleaned.parquet" to the dataframe
# Create a SparkSession
spark = SparkSession.builder.appName("Practice Project").getOrCreate()

df = spark.read.parquet("../datasets/Cleaned/mpg-cleaned.parquet")
rowcount4 = df.count()
print(rowcount4)

df.show(5)
df.printSchema()

# Task 2 : Define the stringindexer pipeline stage
indexer = StringIndexer(inputCol="Origin", outputCol="OriginIndex")

# Task 3 : Define the VectorAssembler Pipeline stage
assembler = VectorAssembler(
    inputCols=[
        "Cylinders",
        "Engine_Disp",
        "Horsepower",
        "Weight",
        "Accelerate",
        "Year",
    ],
    outputCol="features",
)

# Task 4 : Define the StandardScaler Pipeline Stage
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Task 5 : Define the Model Creation Pipeline Stage
lr = LinearRegression(featuresCol="scaledFeatures", labelCol="MPG", regParam=0.01)

# Task 6 : Build the pipeline
pipeline = Pipeline(stages=[indexer, assembler, scaler, lr])

# Task 7 : Splitting the data into two parts (Training Data and Testing Data with seed 42)
(trainingData, testingData) = df.randomSplit([0.7, 0.3], seed=42)

# Task 8 : Fit the pipeline
pipelineModel = pipeline.fit(trainingData)

# Part 2 : Evaluation
print("Part 2 - Evaluation")
print("Total rows =", rowcount4)
ps = [str(x).split("_")[0] for x in pipeline.getStages()]

print("Pipeline Stage 1 =", ps[0])
print("Pipeline Stage 2 =", ps[1])
print("Pipeline Stage 3 =", ps[2])


# Part 3 - Model Evaluation
# Task 1 - Predict using the model
predictions = pipelineModel.transform(testingData)

# Task 2 - Print the MSE (Mean Squared Error)
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    predictionCol="prediction", labelCol="MPG", metricName="mse"
)
mse = evaluator.evaluate(predictions)
print(mse)

# Task 3 - Print the MAE (Mean Absolute Error)
evaluator = RegressionEvaluator(
    predictionCol="prediction", labelCol="MPG", metricName="mae"
)
mae = evaluator.evaluate(predictions)
print(mae)

# Task 4 - Print the R-squared (R2)
evaluator = RegressionEvaluator(
    predictionCol="prediction", labelCol="MPG", metricName="r2"
)
r2 = evaluator.evaluate(predictions)
print(r2)

# Part 3 - Evaluation
print("Part 3 - Evaluation")

print("Mean Squared Error =", round(mse, 2))
print("Mean Absolute Error =", round(mae, 2))
print("R squared (R2) =", round(r2, 2))

lrModel = pipelineModel.stages[-1]

print("Intercept =", round(lrModel.intercept, 2))


# Part 4 - Model Persistence
# Task 1 - Save the model to the path "Practice Project"

pipelineModel.write().overwrite().save("Practice_Project")
# Task 2 - Load the model from the path "Practice_Project"

loadedPipelineModel = PipelineModel.load("Practice_Project")
# Task 3 - Make Predictions using the loaded model on the test data

predictions = loadedPipelineModel.transform(testingData)
# Task 4 - Show the predictions

predictions.select("MPG", "prediction").show()
# Part 4 - Evaluation

loadedmodel = loadedPipelineModel.stages[-1]
totalstages = len(loadedPipelineModel.stages)
inputcolumns = loadedPipelineModel.stages[1].getInputCols()

print("Number of stages in the pipeline =", totalstages)
for i, j in zip(inputcolumns, loadedmodel.coefficients):
    print(f"Coefficient for {i} is {round(j, 4)}")
# Task 5 - Stopping Spark

spark.stop()
