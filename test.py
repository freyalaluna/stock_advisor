from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import numpy as np
from model import evaluate, plot_predictions

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



if __name__ == "__main__":
    actual = np.array([100, 102, 101, 105, 110, 115, 120])
    predicted = np.array([98, 101, 103, 104, 112, 117, 119])

    evaluate(predicted, actual)
    plot_predictions(predicted, actual)