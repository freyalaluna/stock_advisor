from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import numpy as np
from model import *
import torch

def test_spark_rdd():
    sc = SparkContext("local", "test")
    data = [1,2,3,4,5]
    rdd = sc.parallelize(data)
    rdd_sq = rdd.map(lambda x: x ** 2)
    result = rdd_sq.collect()
    print(result)

    sc.stop()

# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName("PySparkTest").master("yarn").getOrCreate()
# data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
# df = spark.createDataFrame(data, ["Name", "Age"])
# df_filtered = df.filter(df.Age > 30)
# result = df_filtered.collect()
#
# # Print the result
# for row in result:
#     print(row)
#
# # Stop the SparkSession
# spark.stop()

def test_StockNN():  
    input_size = 52
    hidden_size = 50
    batch_size = 4

    model = StockNN(input_size, hidden_size)

    x = torch.randn(batch_size, input_size)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

def test_evaluate():
    actual = np.array([109, 102, 108, 101, 110, 124, 125])
    predicted = np.array([98, 101, 103, 104, 111, 117, 119])

    evaluate(predicted, actual)
    
def test_evaluate_constant():
    actual = np.array([5, 5, 5, 5])
    predicted = np.array([4, 6, 5, 7])
    print("\nConstant actual case: ")
    evaluate(predicted, actual)

def test_evaluate_length_mismatch():
    actual = np.array([100, 101, 102])
    predicted = np.array([99, 100])
    print("\nLength mismatch case:")
    try:
        evaluate(predicted, actual)
    except Exception as e:
        print("Get exception as expected:", e)
        
def test_plot_prediction():
    actual = np.array([50, 53, 55, 59, 110, 124, 125])
    predicted = np.array([51, 58, 59, 62, 135, 138, 139])

    plot_predictions(predicted, actual)


    
test_spark_rdd()
test_StockNN()
test_evaluate()
test_evaluate_constant()
test_evaluate_length_mismatch() #should throw exception
test_plot_prediction()