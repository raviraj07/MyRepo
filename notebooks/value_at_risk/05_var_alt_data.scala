// Databricks notebook source
// MAGIC %md
// MAGIC # Value at risk - alternative data
// MAGIC 
// MAGIC **Modernizing risk management practice**: *Traditional banks relying on on-premises infrastructure can no longer effectively manage risk. Banks must abandon the computational inefficiencies of legacy technologies and build an agile Modern Risk Management practice capable of rapidly responding to market and economic volatility. Using value-at-risk use case, you will learn how Databricks is helping FSIs modernize their risk management practices, leverage Delta Lake, Apache Spark and MLFlow to adopt a more agile approach to risk management.*
// MAGIC 
// MAGIC ---
// MAGIC + <a href="$./00_var_context">STAGE0</a>: Home page
// MAGIC + <a href="$./01_var_market_etl">STAGE1</a>: Using Delta Lake for a curated and a 360 view of your risk portfolio
// MAGIC + <a href="$./02_var_model">STAGE2</a>: Tracking experiments and registering risk models through MLflow capabilities
// MAGIC + <a href="$./03_var_monte_carlo">STAGE3</a>: Leveraging the power of Apache Spark for massively distributed Monte Carlo simulations
// MAGIC + <a href="$./04_var_aggregation">STAGE4</a>: Slicing and dicing through your risk exposure using collaborative notebooks and SQL
// MAGIC + <a href="$./05_var_alt_data">STAGE5</a>: Acquiring news analytics data as a proxy of market volatility
// MAGIC + <a href="$./06_var_backtesting">STAGE6</a>: Reporting breaches through model risk backtesting
// MAGIC ---
// MAGIC <antoine.amend@databricks.com>

// COMMAND ----------

// MAGIC %md
// MAGIC ## Context
// MAGIC This notebook augments our risk data with alternative data from news articles. We use [GDELT](https://www.gdeltproject.org/) for that purpose, and focus on every single news event happing in Latin America (in line with our hypothetical portfolio). This demonstrate the potential of alternative data to be 1) more descriptive and potentially 2) predictive by being able to incorporate these events back in the modeling phase, generating market shocks based on actual events as they unfold.

// COMMAND ----------

// MAGIC %md
// MAGIC ## `STEP0` Configuration

// COMMAND ----------

// val news_table_bronze = "antoine_fsi.ws_news_analytics_bronze"
// val news_table_silver = "antoine_fsi.ws_news_analytics_silver"
// val news_table_gold = "antoine_fsi.ws_news_analytics"

val news_table_bronze = "ws_news_analytics_bronze"
val news_table_silver = "ws_news_analytics_silver"
val news_table_gold = "ws_news_analytics"

// COMMAND ----------

// MAGIC %md
// MAGIC ## `STEP1` Access raw data

// COMMAND ----------

// DBTITLE 1,Retrieve GDELT files for year to date
// MAGIC %sh
// MAGIC 
// MAGIC MASTER_URL=http://data.gdeltproject.org/gdeltv2/masterfilelist.txt
// MAGIC 
// MAGIC if [[ -e /tmp/gdelt ]] ; then
// MAGIC   rm -rf /tmp/gdelt
// MAGIC fi
// MAGIC mkdir /tmp/gdelt
// MAGIC 
// MAGIC echo "Retrieve latest URL from [${MASTER_URL}]"
// MAGIC URLS=`curl ${MASTER_URL} 2>/dev/null | awk '{print $3}' | grep gkg.csv.zip | grep gdeltv2/202005011000`
// MAGIC for URL in $URLS; do
// MAGIC   echo "Downloading ${URL}"
// MAGIC   wget $URL -O /tmp/gdelt/gdelt.csv.zip > /dev/null 2>&1
// MAGIC   unzip /tmp/gdelt/gdelt.csv.zip -d /tmp/gdelt/ > /dev/null 2>&1
// MAGIC   LATEST_FILE=`ls -1rt /tmp/gdelt/*.csv | head -1`
// MAGIC   LATEST_NAME=`basename ${LATEST_FILE}`
// MAGIC   cp $LATEST_FILE /dbfs/tmp/gdelt/$LATEST_NAME
// MAGIC   rm -rf /tmp/gdelt/gdelt.csv.zip
// MAGIC   rm $LATEST_FILE
// MAGIC done

// COMMAND ----------

// MAGIC %md
// MAGIC ## `STEP2` Bronze, silver and gold tables

// COMMAND ----------

// DBTITLE 1,Store clean content on Bronze
import com.aamend.spark.gdelt._
val gdeltDF = spark.read.gdeltGkgV2("/tmp/gdelt")
gdeltDF.write.format("delta").mode("overwrite").saveAsTable(news_table_bronze)
display(gdeltDF)

// COMMAND ----------

import com.aamend.spark.gdelt.ContentFetcher

val contentFetcher = new ContentFetcher()
  .setInputCol("documentIdentifier")
  .setOutputContentCol("content")

val contentDF = contentFetcher.transform(gdeltDF)
display(contentDF.select("content"))

// COMMAND ----------

// DBTITLE 1,Filter events on Silver
import org.apache.spark.sql.functions._

val countryCodes = Map(
  "CI" -> "CHILE", 
  "CO" -> "COLUMBIA", 
  "MX" -> "MEXICO", 
  "PM" -> "PANAMA", 
  "PE" -> "PERU"
)

val to_country = udf((s: String) => countryCodes(s))
val filter_theme = udf((s: String) => {
  !s.startsWith("TAX") && !s.matches(".*\\d.*")
})

// Select countries and themes of interest
//spark.readStream.table(news_table_bronze)
spark.read.table(news_table_bronze)
  .withColumn("location", explode(col("locations"))).drop("locations")
  .filter(col("location.countryCode").isin(countryCodes.keys.toArray: _*))
  .withColumn("tone", col("tone.tone"))
  .withColumn("fips", col("location.countryCode"))
  .withColumn("country", to_country(col("fips")))
  .filter(size(col("themes")) > 0)
  .withColumn("theme", explode(col("themes")))
  .filter(filter_theme(col("theme")))
  .select(
    "publishDate",
    "theme",
    "tone",
    "country"
  )
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(news_table_silver)

display(spark.read.table(news_table_silver))

// COMMAND ----------

// DBTITLE 1,Detect trends on Gold table
import java.sql.Timestamp
import java.sql.Date
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

val to_day = udf((s: Timestamp) => new Date(s.getTime()))
val to_time = udf((s: Date) => s.getTime() / 1000)

// Group themes by day and country, finding total number of articles
val dailyThemeDF = spark.read.table(news_table_silver)
  .withColumn("day", to_day(col("publishDate")))
  .groupBy("day", "country", "theme")
  .agg(
    sum(lit(1)).as("total"),
    avg(col("tone")).as("tone")
  )
  .filter(col("day") >= "2020-01-01")
  .select("day", "country", "theme", "total", "tone")

// Use total number of articles by country
val dailyCountryDf = dailyThemeDF
  .groupBy("day", "country")
  .agg(sum(col("total")).as("global"))

// Normalize number of articles as proxy for media coverage
val mediaCoverageDF = dailyCountryDf
  .join(dailyThemeDF, List("country", "day"))
  .withColumn("coverage", lit(100) * col("total") / col("global"))
  .select("day", "country", "theme", "coverage", "tone", "total")

// Detect trends using a cross over between a 7 and a 30 days window
val ma30 = Window.partitionBy("country", "theme").orderBy("time").rangeBetween(-30 * 24 * 3600, 0)
val ma07 = Window.partitionBy("country", "theme").orderBy("time").rangeBetween(-7 * 24 * 3600, 0)

// Detect trends
val trendDF = mediaCoverageDF
  .withColumn("time", to_time(col("day")))
  .withColumn("ma30", avg(col("coverage")).over(ma30))
  .withColumn("ma07", avg(col("coverage")).over(ma07))
  .select("day", "country", "theme", "coverage", "ma07", "ma30", "tone", "total")
  .orderBy(asc("day"))
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(news_table_gold)

display(spark.read.table(news_table_gold))

// COMMAND ----------

// MAGIC %md
// MAGIC ## `STEP3` Detect trends

// COMMAND ----------

// DBTITLE 1,Get highest media coverage per country in last quarter
// MAGIC %sql
// MAGIC CREATE OR REPLACE TEMPORARY VIEW grouped_news AS 
// MAGIC SELECT 
// MAGIC   country, 
// MAGIC   theme, 
// MAGIC   MAX(coverage) AS moving_average,
// MAGIC   COUNT(1) AS total
// MAGIC FROM ws_news_analytics
// MAGIC WHERE LENGTH(theme) > 0
// MAGIC GROUP BY theme, country;
// MAGIC   
// MAGIC SELECT * FROM (
// MAGIC   SELECT *, row_number() OVER (PARTITION BY country ORDER BY moving_average DESC) rank 
// MAGIC   FROM grouped_news
// MAGIC ) tmp
// MAGIC WHERE rank <= 5
// MAGIC ORDER BY country

// COMMAND ----------

// DBTITLE 1,Timeline events for mining activities in Peru
// MAGIC %sql
// MAGIC SELECT 
// MAGIC   day,
// MAGIC   total,
// MAGIC   ma30,
// MAGIC   ma07,
// MAGIC   CASE 
// MAGIC     WHEN sig = 0 THEN 'N/A'
// MAGIC     WHEN sig = -1 THEN 'HIGH'
// MAGIC     WHEN sig = 1 THEN 'LOW'
// MAGIC   END AS trend
// MAGIC FROM (
// MAGIC   SELECT day, total, ma30, ma07, SIGNUM(ma30 - ma07) AS sig FROM ws_news_analytics
// MAGIC   WHERE country = "PERU"
// MAGIC   AND theme = 'ENV_MINING'
// MAGIC ) tmp
// MAGIC WHERE sig != 0
// MAGIC ORDER BY day ASC

// COMMAND ----------

// MAGIC %md
// MAGIC ## `HOMEWORK` Predict market volatility
// MAGIC 
// MAGIC With news analytics available on delta, could you predict market volatility based on news events, as they unfold?
// MAGIC 
// MAGIC <img src="/files/shared_uploads/finserv/images/volatility_news.png" alt="logical_flow" width="500">
// MAGIC 
// MAGIC Reference: [https://eprints.soton.ac.uk/417880/1/manuscript2_2.pdf](https://eprints.soton.ac.uk/417880/1/manuscript2_2.pdf)

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC + <a href="$./00_var_context">STAGE0</a>: Home page
// MAGIC + <a href="$./01_var_market_etl">STAGE1</a>: Using Delta Lake for a curated and a 360 view of your risk portfolio
// MAGIC + <a href="$./02_var_model">STAGE2</a>: Tracking experiments and registering risk models through MLflow capabilities
// MAGIC + <a href="$./03_var_monte_carlo">STAGE3</a>: Leveraging the power of Apache Spark for massively distributed Monte Carlo simulations
// MAGIC + <a href="$./04_var_aggregation">STAGE4</a>: Slicing and dicing through your risk exposure using collaborative notebooks and SQL
// MAGIC + <a href="$./05_var_alt_data">STAGE5</a>: Acquiring news analytics data as a proxy of market volatility
// MAGIC + <a href="$./06_var_backtesting">STAGE6</a>: Reporting breaches through model risk backtesting
// MAGIC ---

// COMMAND ----------

