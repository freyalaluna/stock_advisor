import filterNews
import filterStocks
import joinStocksWithNews
import model
from pyspark.sql import SparkSession

targetStocks = ["GOOGL", "AAPL", "FB", "NVDA", "AMZN", "MSFT", "TSLA"]
files = ["/StockAdvisor/datasets/Stocks/AMZN.csv", "/StockAdvisor/datasets/Stocks/AAPL.csv", "/StockAdvisor/datasets/Stocks/FB.csv",
         "/StockAdvisor/datasets/Stocks/NVDA.csv", "/StockAdvisor/datasets/Stocks/MSFT.csv", "/StockAdvisor/datasets/Stocks/GOOGL.csv",
         "/StockAdvisor/datasets/Stocks/TSLA.csv"]

def main() -> None:
    spark = (SparkSession.builder
             .appName("FilterNews")
             .master("yarn")
             .getOrCreate())

    # filterNews.execute(targetStocks)
    # filterStocks.execute(files)
    joinStocksWithNews.execute()
    model.execute()

    spark.stop()



if __name__ == '__main__':
    main()