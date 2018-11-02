from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col,monotonically_increasing_id
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import desc
import pyspark.sql.functions as psf
from pyspark.sql import SQLContext

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
sqlContext = SQLContext(sc)
spark.conf.set("spark.sql.shuffle.partitions",10000)
 
df = spark.read.parquet("/user/lsde22/out_all_fe100.prq")
df = df.select(col("doi"),col("norm_tfidf").alias("features")).repartition(10000)

# Trains a KMeans model.
kmeans = KMeans().setK(1000).setMaxIter(10)
km_model = kmeans.fit(df)

clustersTable = km_model.transform(df).repartition(10000)
df_pred = clustersTable.select('doi', 'features','prediction').repartition(10000)
df_pred = df_pred.withColumn("ID",monotonically_increasing_id()).repartition(10000)


dot_udf = psf.udf(lambda x,y: float(x.dot(y))/float(x.norm(2) * y.norm(2)), DoubleType())
data = df_pred.alias("i").join(df_pred.alias("j"), (psf.col("i.ID") < psf.col("j.ID")) & (psf.col("i.prediction") == psf.col("j.prediction"))).select(psf.col("i.doi").alias("doi_i"), psf.col("j.doi").alias("doi_j"),dot_udf(psf.col("i.features"),psf.col("j.features")).alias("similarity")).sort(desc("similarity"))
data = data.repartition(10000)

top = data.limit(5000)
top.write.parquet("/user/lsde22/sim_pairs_fe100")
