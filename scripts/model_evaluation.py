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
