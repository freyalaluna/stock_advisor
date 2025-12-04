from pyspark.sql import SparkSession
import numpy as np
import torch
import torch.nn as nn


class StockNN(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(StockNN, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, 1)
  
  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x

"""
This assumes that the data arrives in a spark dataframe with
the labels attached as listed. I doubt this is the case, so
consider this entire method a TODO
"""
def prepare_data(data):
  df = data.toPandas()
  
  X_list = []
  y_list = []
  
  for i in range(len(df)):
    row = df.iloc[i]
    
    stock_highs = row['stock_highs_30d']
    sentiment = row['news_sentiment_5d'] 
    confidence = row['news_confidence_5d']
    variance = row['stock_var']

    sentiment_list = list(sentiment)
    confidence_list = list(confidence)
    if(len(sentiment_list) < 10):
      for i in range(10-len(sentiment_list)): sentiment_list.append(0)
      for i in range(10-len(confidence_list)): confidence_list.append(0.0)

    
    # just flatten to a feature list since not LSTM anymore
    features = list(stock_highs) + sentiment_list + confidence_list + [variance]
    
    X_list.append(features)
    y_list.append(row['target_price_7d'])
  
  X = np.array(X_list)
  y = np.array(y_list)
  
  return X, y


def train(partition_data, model_state, epochs_per_partition):
  X, y = partition_data
  
  X_tensor = torch.FloatTensor(X)
  y_tensor = torch.FloatTensor(y)
  
  model = StockNN(41, 50)
  model.load_state_dict(model_state)
  
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
  for epoch in range(epochs_per_partition):
    model.train()
    
    outputs = model(X_tensor)
    loss = criterion(outputs.squeeze(), y_tensor)
    
    # This is where we can add custom back prop if we decide that's necessary later
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
      print(f'Epoch [{epoch+1}/{epochs_per_partition}], Loss: {loss.item():.3f}')
  
  return model.state_dict()


def combine_models(states):
  combined = dict()
  
  for key in states[0].keys():
    combined[key] = torch.stack([state[key] for state in states]).mean(dim=0)
  
  return combined


def distributed_train(data, num_partitions, epochs, epochs_per_partition):
  X, y = prepare_data(data)
  
  X_partitions = np.array_split(X, num_partitions)
  y_partitions = np.array_split(y, num_partitions)
  partitioned_data = list(zip(X_partitions, y_partitions))
  
  model = StockNN(41, 50)
  state = model.state_dict()
  
  sc = data.rdd.context
  
  for _ in range(epochs):

    broadcasted = sc.broadcast(state)                                           # send the updated weights to every training task
    partitions_rdd = sc.parallelize(partitioned_data, num_partitions)           # convert the partitions of data into an RDD
    updated_states = partitions_rdd.map(                                        # train on each partition with a different spark task
        lambda p: train(p, broadcasted.value, epochs_per_partition)
    ).collect()
    
    state = combine_models(updated_states)                                      # rejoin the weights
  
  out = StockNN(41, 50)                                                         # createa. new model to hold the final weights (can be used for testing)
  out.load_state_dict(state)                                                    # load the weights to the model
  
  return out


def predict(model, data):
  X, y = prepare_data(data)
  X = torch.FloatTensor(X)
  
  model.eval()
  predictions = model(X)
  
  return predictions.detach().numpy().flatten()

def execute():
    
  spark = SparkSession.builder.getOrCreate()

  df = spark.read.parquet("/StockAdvisor/datasets/filtered/stocksWithSentiments").orderBy("ticker")
  # for i in range(1000):
  #   data.append({
  #     'ticker': f'STOCK{i}',
  #     'stock_highs_30d': [float(x) for x in np.random.randn(30)],
  #     'news_sentiment_5d': [float(x) for x in np.random.randn(5)],
  #     'news_confidence_5d': [float(x) for x in np.random.rand(5)],
  #     'stock_var': float(np.random.rand()),
  #     'target_price_7d': float(np.random.randn())
  #   })
  # df = spark.createDataFrame(data)
  
  train_df, test_df = df.randomSplit([0.8, 0.2])
  
  print(f"Train size: {train_df.count()}")
  print(f"Test size: {test_df.count()}")
  
  num_partitions = 5
  epochs = 10
  epochs_per_partition = 5
  model = distributed_train(train_df, num_partitions, epochs, epochs_per_partition)
  
  predictions = predict(model, test_df)
  
  print(f"Num predictions: {len(predictions)}")
  
  # spark.stop()\
  
