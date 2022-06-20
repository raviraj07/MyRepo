# Databricks notebook source
# MAGIC %md
# MAGIC # Value at risk - create portfolio
# MAGIC 
# MAGIC **Modernizing risk management practice**: *Traditional banks relying on on-premises infrastructure can no longer effectively manage risk. Banks must abandon the computational inefficiencies of legacy technologies and build an agile Modern Risk Management practice capable of rapidly responding to market and economic volatility. Using value-at-risk use case, you will learn how Databricks is helping FSIs modernize their risk management practices, leverage Delta Lake, Apache Spark and MLFlow to adopt a more agile approach to risk management.*
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./00_var_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_var_market_etl">STAGE1</a>: Using Delta Lake for a curated and a 360 view of your risk portfolio
# MAGIC + <a href="$./02_var_model">STAGE2</a>: Tracking experiments and registering risk models through MLflow capabilities
# MAGIC + <a href="$./03_var_monte_carlo">STAGE3</a>: Leveraging the power of Apache Spark for massively distributed Monte Carlo simulations
# MAGIC + <a href="$./04_var_aggregation">STAGE4</a>: Slicing and dicing through your risk exposure using collaborative notebooks and SQL
# MAGIC + <a href="$./05_var_alt_data">STAGE5</a>: Acquiring news analytics data as a proxy of market volatility
# MAGIC + <a href="$./06_var_backtesting">STAGE6</a>: Reporting breaches through model risk backtesting
# MAGIC ---
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context
# MAGIC In this notebook, we use `yfinance` to download stock data for 40 equities in an hypothetical Latin America portfolio. We show how to use `pandas UDF` paradigm to distribute this process efficiently and store all of our output data as a **Delta Lake** table so that our data is analytic ready at every point in time.

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP0` Configuration

# COMMAND ----------

# DBTITLE 1,Import statements
import yfinance as yf
import pandas as pd
import numpy as np
from io import StringIO
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import udf
from pyspark.sql.functions import pandas_udf, PandasUDFType
from datetime import datetime, timedelta

# COMMAND ----------

# DBTITLE 1,Control parameters
# portfolio_table = 'antoine_fsi.ws_portfolio'
# stock_table = 'antoine_fsi.ws_stock'
# stock_return_table = 'antoine_fsi.ws_stock_return'
# market_table = 'antoine_fsi.ws_market'
# market_return_table = 'antoine_fsi.ws_market_return'

portfolio_table = 'ws_portfolio'
stock_table = 'ws_stock'
stock_return_table = 'ws_stock_return'
market_table = 'ws_market'
market_return_table = 'ws_market_return'

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1` Create our portfolio

# COMMAND ----------

# DBTITLE 1,Define our portfolio
portfolio = """
country,company,ticker,industry
CHILE,Banco de Chile,BCH,Banks
CHILE,Banco Santander-Chile,BSAC,Banks
CHILE,Compañía Cervecerías Unidas S.A.,CCU,Beverages
CHILE,Itaú CorpBanca,ITCB,Banks
CHILE,"Embotelladora Andina, S.A.",AKOA,Beverages
CHILE,"Embotelladora Andina, S.A.",AKOB,Beverages
CHILE,"Empresa Nacional de Electricidad, S.A. (Chile)",EOCC,Electricity
CHILE,"Enersis, S.A.",ENIA,Electricity
CHILE,Enersis Chile SA Sponsored ADR,ENIC,Electricity
CHILE,LAN Airlines S.A.,LFL,Travel & Leisure
CHILE,"SQM-Sociedad Química y Minera de Chile, S.A.",SQM,Chemicals
CHILE,"Viña Concha y Toro, S.A.",VCO,Beverages
COLOMBIA,Avianca Holdings S.A.,AVH,Travel & Leisure
COLOMBIA,BanColombia S.A.,CIB,Banks
COLOMBIA,Ecopetrol S.A.,EC,Oil & Gas Producers
COLOMBIA,Grupo Aval Acciones y Valores S.A,AVAL,Financial Services
MEXICO,"América Móvil, S.A.B. de C.V.",AMX,Mobile Telecommunications
MEXICO,América Móvil SAB de CV Sponsored ADR Class A,AMOV,Mobile Telecommunications
MEXICO,CEMEX S.A.B. de C.V. (CEMEX),CX,Construction & Materials
MEXICO,"Coca-Cola FEMSA, S.A.B. de C.V.",KOF,Beverages
MEXICO,"Controladora Vuela Compañía de Aviación, S.A.B. de C.V",VLRS,Travel & Leisure
MEXICO,"Fomento Económico Mexicano, S.A.B. de C.V. (FEMSA)",FMX,Beverages
MEXICO,"Grupo Aeroportuario del Pacífico, S.A.B. de C.V. (GAP)",PAC,Industrial Transportation
MEXICO,"Grupo Aeroportuario del Sureste, S.A. de C.V. (ASUR)",ASR,Industrial Transportation
MEXICO,"Grupo Financiero Santander México, S.A.B. de C.V",BSMX,Banks
MEXICO,"Grupo Simec, S.A. De CV. (ADS)",SIM,Industrial Metals & Mining
MEXICO,"Grupo Televisa, S.A.",TV,Media
MEXICO,"Industrias Bachoco, S.A.B. de C.V. (Bachoco)",IBA,Food Producers
PANAMA,"Banco Latinoamericano de Comercio Exterior, S.A.",BLX,Banks
PANAMA,"Copa Holdings, S.A.",CPA,Travel & Leisure
PERU,Cementos Pacasmayo S.A.A.,CPAC,Construction & Materials
PERU,Southern Copper Corporation,SCCO,Industrial Metals & Mining
PERU,Fortuna Silver Mines Inc.,FSM,Mining
PERU,Compañía de Minas Buenaventura S.A.,BVN,Mining
PERU,Graña y Montero S.A.A.,GRAM,Construction & Materials
PERU,Credicorp Ltd.,BAP,Banks
"""

portfolio_df = pd.read_csv(StringIO(portfolio))

# COMMAND ----------

spark.sql("use default")

# COMMAND ----------

# DBTITLE 1,Store our portfolio
spark \
  .createDataFrame(portfolio_df) \
  .select('ticker', 'company', 'country', 'industry') \
  .write \
  .format('delta') \
  .mode('overwrite') \
  .saveAsTable(portfolio_table)

display(spark.read.table(portfolio_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP2` Download stock data

# COMMAND ----------

# DBTITLE 1,Download tick data from yahoo finance
 schema = StructType(
  [
    StructField('ticker', StringType(), True), 
    StructField('date', DateType(), True),
    StructField('open', DoubleType(), True),
    StructField('high', DoubleType(), True),
    StructField('low', DoubleType(), True),
    StructField('close', DoubleType(), True),
    StructField('volume', DoubleType(), True),
  ]
)

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def fetch_tick(group, pdf):
  tick = group[0]
  try:
    msft = yf.Ticker(tick)
    raw = msft.history(period="2y")[['Open', 'High', 'Low', 'Close', 'Volume']]
    # fill in missing business days
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq='B')
    # use last observation carried forward for missing value
    output_df = raw.reindex(idx, method='pad')
    # Pandas does not keep index (date) when converted into spark dataframe
    output_df['date'] = output_df.index
    output_df['ticker'] = tick    
    output_df = output_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
    return output_df
  except:
    return pd.DataFrame(columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])
  
spark \
  .read \
  .table(portfolio_table) \
  .groupBy("ticker") \
  .apply(fetch_tick) \
  .write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable(stock_table)

display(spark.read.table(stock_table))

# COMMAND ----------

# DBTITLE 1,Create widget to access specific instrument
# dbutils.widgets.remove('stock')
tickers = spark.read.table(portfolio_table).select('ticker').toPandas()['ticker']
dbutils.widgets.dropdown('stock', 'AVAL', tickers)

# COMMAND ----------

# DBTITLE 1,Access a specific instrument
display(
  spark \
    .read \
    .table(stock_table) \
    .filter(F.col('ticker') == dbutils.widgets.get('stock')) \
    .orderBy(F.asc('date'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP3` Download market factors

# COMMAND ----------

factors = {
  '^GSPC':'SP500',
  '^NYA':'NYSE',
  '^XOI':'OIL',
  '^TNX':'TREASURY',
  '^DJI':'DOWJONES'
}

# Create a pandas dataframe where each column contain close index
factors_df = pd.DataFrame()
for tick in factors.keys():    
    msft = yf.Ticker(tick)
    raw = msft.history(period="2y")
    # fill in missing business days
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq='B')
    # use last observation carried forward for missing value
    pdf = raw.reindex(idx, method='pad')
    factors_df[factors[tick]] = pdf['Close'].copy()
        
# Pandas does not keep index (date) when converted into spark dataframe
factors_df['Date'] = idx

# Overwrite delta table (bronze) with information to date
spark.createDataFrame(factors_df) \
  .write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable(market_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP4` Compute daily log return

# COMMAND ----------

# DBTITLE 1,Daily log return of market factors
# our market factors easily fit in memory, use pandas for convenience
df = spark.table(market_table).toPandas()

# add date column as pandas index for sliding window
df.index = df['Date']
df = df.drop(columns = ['Date'])

# compute daily log returns
df = np.log(df.shift(1)/df)

# add date columns
df['date'] = df.index

# overwrite log returns to market table (gold)
spark.createDataFrame(df) \
  .write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable(market_return_table)

# COMMAND ----------

# DBTITLE 1,Daily log returns of instruments
# Create UDF for computing daily log returns
@udf("double")
def compute_return(first, close):
  return float(np.log(close / first))

# Apply a tumbling 1 day window on each instrument
window = Window.partitionBy('ticker').orderBy('date').rowsBetween(-1, 0)

# apply sliding window and take first element
# compute returns
# make sure we have corresponding dates in market factor tables
sdf = spark.table(stock_table) \
  .filter(F.col('close').isNotNull()) \
  .withColumn("first", F.first('close').over(window)) \
  .withColumn("return", compute_return('first', 'close')) \
  .select('date', 'ticker', 'return') \
  .join(spark.table(market_return_table), 'date') \
  .select('date', 'ticker', 'return')

# overwrite log returns to market table (gold)
sdf.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable(stock_return_table)

# COMMAND ----------

# DBTITLE 1,Distribution of returns
display(spark.table(stock_return_table).filter(F.col('ticker') == dbutils.widgets.get('stock')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP5` Ensure data consistency

# COMMAND ----------

# DBTITLE 1,Access audit history
# MAGIC %sql
# MAGIC DESCRIBE HISTORY ws_market_return

# COMMAND ----------

# DBTITLE 1,Access data at a specific version
# MAGIC %sql
# MAGIC SELECT * FROM ws_market_return
# MAGIC TIMESTAMP AS OF '2022-06-17 22:12:09'

# COMMAND ----------

# MAGIC %md
# MAGIC ## `HOMEWORK` Unifying streaming and batch
# MAGIC 
# MAGIC Can you read new market data as stream using `.readStream` and `Trigger.ONCE` so that only delta is processed?
# MAGIC 
# MAGIC ```
# MAGIC val inputStream = spark
# MAGIC   .readStream
# MAGIC   .format("delta")
# MAGIC   .table("SILVER_TABLE")
# MAGIC 
# MAGIC val outputStream = inputStream.doSomething()
# MAGIC 
# MAGIC outputStream
# MAGIC   .writeStream
# MAGIC   .trigger(Trigger.Once)
# MAGIC   .option("checkpointLocation", "/my/checkpoint/dir")
# MAGIC   .format("delta")
# MAGIC   .table("GOLD_TABLE")
# MAGIC   ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC + <a href="$./00_var_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_var_market_etl">STAGE1</a>: Using Delta Lake for a curated and a 360 view of your risk portfolio
# MAGIC + <a href="$./02_var_model">STAGE2</a>: Tracking experiments and registering risk models through MLflow capabilities
# MAGIC + <a href="$./03_var_monte_carlo">STAGE3</a>: Leveraging the power of Apache Spark for massively distributed Monte Carlo simulations
# MAGIC + <a href="$./04_var_aggregation">STAGE4</a>: Slicing and dicing through your risk exposure using collaborative notebooks and SQL
# MAGIC + <a href="$./05_var_alt_data">STAGE5</a>: Acquiring news analytics data as a proxy of market volatility
# MAGIC + <a href="$./06_var_backtesting">STAGE6</a>: Reporting breaches through model risk backtesting
# MAGIC ---

# COMMAND ----------

