# from pyspark.sql import SparkSession
# from pyspark import SparkConf, SparkContext
#
# sc = SparkContext("local", "test")
# data = [1,2,3,4,5]
# rdd = sc.parallelize(data)
# rdd_sq = rdd.map(lambda x: x ** 2)
# result = rdd_sq.collect()
#
# print(result)
#
# sc.stop()

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PySparkTest").master("local").getOrCreate()
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df = spark.createDataFrame(data, ["Name", "Age"])
df_filtered = df.filter(df.Age > 30)
result = df_filtered.collect()

# Print the result
for row in result:
    print(row)

# Stop the SparkSession
spark.stop()