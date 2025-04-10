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
