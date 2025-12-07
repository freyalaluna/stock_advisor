from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import numpy as np
from model import *
import torch
import matplotlib.pyplot as plt

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

def test_StockNN_singlesample():
    input_size = 52
    hidden_size = 50
    batch_size = 1

    model = StockNN(input_size, hidden_size)
    x = torch.randn(batch_size, input_size)
    output = model(x)

    print("\nSingle sample case:")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

def test_StockNN_large():
    input_size = 69
    hidden_size = 50
    batch_size = 129

    model = StockNN(input_size, hidden_size)
    x = torch.randn(batch_size, input_size)
    output = model(x)

    print("\nLarge batch case:")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    
def test_prepare_data():
    print("\nprepare data:")

    spark = SparkSession.builder.master("local[*]").appName("prepareDataTest").getOrCreate()

    data = [
        {
            "stock_highs_30d": [100] * 30,
            "news_sentiment_5d": [1, 0, -1, 1, 0],
            "news_confidence_5d": [0.9, 0.8, 0.85, 0.7, 0.6],
            "stock_var": 1.5,
            "target_price_7d": 110.0,
        }
    ]

    df = spark.createDataFrame(data)
    X, y = prepare_data(df)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    spark.stop()
    
def test_prepare_data_empty():
    print("\nprepare data empty:")

    spark = SparkSession.builder.master("local[*]").appName("prepareDataEmptyTest").getOrCreate()


    try:
        df = spark.createDataFrame([], schema="")
        X, y = prepare_data(df)
        print("X shape:", X.shape)
        print("y shape:", y.shape)
    except Exception as e:
        print("Got exception:", e)

    spark.stop()

def test_prepare_datamissing():
    print("\nprepare data missing:")

    spark = SparkSession.builder.master("local[*]").appName("prepareDataPaddingTest").getOrCreate()

    data = [
        {
            "stock_highs_30d": [95] * 30,
            "news_sentiment_5d": [],
            "news_confidence_5d": None,
            "stock_var": 2.1,
            "target_price_7d": 98.0,
        }
    ]

    try:
        df = spark.createDataFrame(data)
        X, y = prepare_data(df)
    except Exception as e:
        print("Got exception as expected:", e)
    spark.stop()

def test_train():
    X = np.random.rand(10, 52)
    y = np.random.rand(10)
    partition_data = (X, y)

    model = StockNN(52, 50)
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    new_state = train(partition_data, model.state_dict(), epochs_per_partition=2)

    key = list(initial_state.keys())[0]

    before = initial_state[key].flatten()[:5]
    after = new_state[key].flatten()[:5]
    diff = after - before

    print("First 5 values before:")
    print(before)

    print("First 5 values after:")
    print(after)

    print("First 5 value dif:")
    print(diff)
    
def test_train_zero():
    X = np.random.rand(10, 52)
    y = np.random.rand(10)
    partition_data = (X, y)

    model = StockNN(52, 50)
    state = {k: v.clone() for k, v in model.state_dict().items()}

    new_state = train(partition_data, model.state_dict(), epochs_per_partition=0)

    key = list(state.keys())[0]
    diff = new_state[key] - state[key]

    print("Zero diff:")
    print(diff.abs().sum())


def test_train_single():
    X = np.random.rand(1, 52)
    y = np.random.rand(1)
    partition_data = (X, y)

    model = StockNN(52, 50)
    state = {k: v.clone() for k, v in model.state_dict().items()}

    new_state = train(partition_data, model.state_dict(), epochs_per_partition=2)

    key = list(state.keys())[0]

    before = state[key].flatten()[:5]
    after = new_state[key].flatten()[:5]
    diff = after - before

    print("Single sample before:")
    print(before)

    print("Single sample after:")
    print(after)

    print("Single sample diff:")
    print(diff)

def test_combine_models_normal():
    print("combine models :")

    model1 = StockNN(52, 50).state_dict()
    model2 = StockNN(52, 50).state_dict()
    model3 = StockNN(52, 50).state_dict()

    combined = combine_models([model1, model2, model3])

    key = list(model1.keys())[0]

    print("Original first values:")
    print(model1[key].flatten()[:3])

    print("Combined first values:")
    print(combined[key].flatten()[:3])


def test_combine_models_two():
    print("\ncombine models two:")

    model1 = StockNN(52, 50).state_dict()
    model2 = StockNN(52, 50).state_dict()

    combined = combine_models([model1, model2])

    key = list(model1.keys())[0]

    diff = combined[key] - ((model1[key] + model2[key]) / 2)

    print("Difference sum:")
    print(diff.abs().sum().item())


def test_combine_empty():
    print(" combine empty:")

    try:
        combine_models([])
    except Exception as e:
        print("Got exception as expected:", e)
        
def make_df_52(spark, n):
    data = []
    for i in range(n):
        row = {
            "stock_highs_30d": [100 + i] * 31,
            "news_sentiment_5d": [1, 0, -1, 1, 0],
            "news_confidence_5d": [0.9, 0.8, 0.85, 0.7, 0.6],
            "stock_var": 1.5,
            "target_price_7d": 110.0,
        }
        data.append(row)
    df = spark.createDataFrame(data)
    return df

def test_distributed_train():
    print("\ndistributed train normal")

    spark = SparkSession.builder.master("local[*]").appName("distributedTrainNormal").getOrCreate()
    df = make_df_52(spark, 4)

    model = distributed_train(df, num_partitions=2, epochs=1, epochs_per_partition=1)

    x = torch.randn(2, 52)
    out = model(x)

    print("Output shape:", out.shape)

    spark.stop()


def test_distributed_train_zero():
    print("\ndistributed train zero")

    spark = SparkSession.builder.master("local[*]").appName("distributedTrainZero").getOrCreate()
    df = make_df_52(spark, 4)

    model = distributed_train(df, num_partitions=2, epochs=0, epochs_per_partition=2)

    x = torch.randn(1, 52)
    out = model(x)

    print("Output shape:", out.shape)

    spark.stop()


def test_distributed_train_single():
    print("\ndistributed train single")

    spark = SparkSession.builder.master("local[*]").appName("distributedTrainSingle").getOrCreate()
    df = make_df_52(spark, 1)

    model = distributed_train(df, num_partitions=1, epochs=1, epochs_per_partition=1)

    x = torch.randn(1, 52)
    out = model(x)

    print("Output shape:", out.shape)

    spark.stop()

def test_predict():
    print("\npredict normal")

    spark = SparkSession.builder.master("local[*]").appName("predictNormal").getOrCreate()
    df = make_df_52(spark, 5)

    model = StockNN(52, 50)
    preds, targets = predict(model, df)

    print("Pred length:", len(preds))
    print("Target length:", len(targets))

    spark.stop()


def test_predict_single():
    print("\npredict single row")

    spark = SparkSession.builder.master("local[*]").appName("predictSingle").getOrCreate()
    df = make_df_52(spark, 1)

    model = StockNN(52, 50)
    preds, targets = predict(model, df)

    print("Pred length:", len(preds))
    print("Target value:", targets[0])

    spark.stop()


def test_predict_empty():
    print("\npredict empty")

    spark = SparkSession.builder.master("local[*]").appName("predictEmpty").getOrCreate()

    schema = ""
    df = spark.createDataFrame([], schema=schema)

    model = StockNN(52, 50)

    try:
        preds, targets = predict(model, df)
        print("Pred length:", len(preds))
        print("Target length:", len(targets))
    except Exception as e:
        print("Got exception:", e)

    spark.stop()

def make_execute_df(spark, n):
    data = []
    for i in range(n):
        row = {
            "stock_highs_30d": [100 + i] * 31,
            "news_sentiment_5d": [1, 0, -1, 1, 0],
            "news_confidence_5d": [0.9, 0.8, 0.85, 0.7, 0.6],
            "stock_var": 1.5,
            "target_price_7d": 110.0,
        }
        data.append(row)
    return spark.createDataFrame(data)


def test_execute():
    print("\nexecute test")

    spark = SparkSession.builder.master("local[*]").appName("executeTest").getOrCreate()

    df = make_execute_df(spark, 10).orderBy("target_price_7d")

    train_df, test_df = df.randomSplit([0.8, 0.2])

    train_size = train_df.count()
    test_size = test_df.count()

    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")

    num_partitions = 2
    epochs = 1
    epochs_per_partition = 1

    model = distributed_train(train_df, num_partitions, epochs, epochs_per_partition)

    if test_size > 0:
        predictions, actual = predict(model, test_df)
        print(f"Num predictions: {len(predictions)}")
        print(f"Num actual: {len(actual)}")
    else:
        print("Skip predict because test split is empty")

    spark.stop()



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

    plot_predictions(predicted, actual, filename="test_prediction_plot.png")

def test_plot_single_point():
    actual = np.array([100])
    predicted = np.array([95])

    print("\nSingle point plot:")
    plot_predictions(predicted, actual, filename="plot_single_point.png")

def test_plot_empty():
    actual = np.array([])
    predicted = np.array([])

    print("\nEmpty plot:")
    try:
        plot_predictions(predicted, actual, filename="plot_empty.png")
    except Exception as e:
        print("Got exception", e)


    
#test_spark_rdd()
#test_StockNN()
#test_StockNN_singlesample()
#test_StockNN_large()
#test_prepare_data()
#test_prepare_data_empty()
#test_prepare_datamissing() #should throw exception
#test_train()
#test_train_zero()
#test_train_single()
#test_combine_models_normal()
#test_combine_models_two()
#test_combine_empty()
#test_distributed_train()
#test_distributed_train_zero()
#test_distributed_train_single()
#test_predict()
#test_predict_single()
#test_predict_empty()
#test_execute()
#test_evaluate()
#test_evaluate_constant()
#test_evaluate_length_mismatch() #should throw exception
#test_plot_prediction()
#test_plot_single_point()
#test_plot_empty()
