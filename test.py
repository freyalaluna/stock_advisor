from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

import numpy as np
import matplotlib.pyplot as plt
from lstm import plot_predictions


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
   

def main():
    
  
    actual_prices = np.array([
        100, 102, 101, 103, 105, 107, 106, 108, 110, 111,
        113, 115, 114, 116, 118, 120, 119, 121, 123, 125
    ])

    predicted_prices = np.array([
        99, 101, 102, 104, 104, 108, 107, 107, 111, 112,
        114, 114, 115, 117, 119, 121, 120, 122, 124, 126
    ])

    plot_predictions(predicted_prices, actual_prices)

if __name__ == "__main__":
    main()
