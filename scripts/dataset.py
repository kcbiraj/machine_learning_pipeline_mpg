# Part 1: ETL

# Task 1: Importing required libraries
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


# Suppressing the warnings
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
warnings.filterwarnings("ignore")

# Findspark simplifies the process of using Apache Spark with Python
import findspark

findspark.init()

# Another way of suppressing the warnings
# import warnings
# warnings.filterwarnings('ignore', module = 'sklearn')

# Task 2: Creating Spark Session
spark = SparkSession.builder.appName("Practice Project").getOrCreate()
# Task 3 : Load the CSV file into a dataframe

df = spark.read.csv("../datasets/Raw/mpg-raw.csv", header=True, inferSchema=True)
# Task 4: Print top 5 rows of the dataset

df.show(5)
# Task 5: Print the number of cars in each origin

df.groupBy("Origin").count().orderBy("count").show()
# Task 6: Print the total number of rows in the dataset

rowcount1 = df.count()
print(rowcount1)

# Task 7: Drop all the duplicate rows from the dataset

df = df.dropDuplicates()
df.show()
# Task 8 : Print the total number of rows in the dataset after dropping duplicates

rowcount2 = df.count()
print(rowcount2)
# Task 9 : Drop all the rows that contain null values in the dataset
df = df.dropna()
df.show()

# Task 10 : Printing the total number of rows after dropping null values from the dataset
rowcount3 = df.count()
print(rowcount3)

# Task 11 : Rename the column "Engine Disp" to "Engine_Disp" Drop
df = df.withColumnRenamed("Engine Disp", "Engine_Disp")
df.show()

# Task 12 : Save the dataframe in parquet format, name the file as "mpg-cleaned.parquet"
df.write.mode("overwrite").parquet("../datasets/Cleaned/mpg-cleaned.parquet")

# Part 1 : Evaluation

print("Part 1 - Evaluation")
print("Total rows = ", rowcount1)
print("Total rows after dropping duplicate rows = ", rowcount2)
print(
    "Total rows after dropping duplicate rows and null values from the dataset = ",
    rowcount3,
)
print("Renamed column name =", df.columns[2])
print("mpg-cleaned.parquet exists :", os.path.isdir("mpg-cleaned.parquet"))


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
