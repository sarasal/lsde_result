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

df = spark.read.parquet("/user/lsde22/out_all_lda_10topics_100idf.prq")
df = df.drop("features").withColumnRenamed("topicDistribution","features").repartition(10000)

# Trains a KMeans model.
kmeans = KMeans().setK(20).setMaxIter(10)
km_model = kmeans.fit(df)

clustersTable = km_model.transform(df).repartition(10000)
pred = clustersTable.select('doi', 'norm_tfidf','prediction').repartition(10000)

g0= pred.filter('prediction==0')
g1= pred.filter('prediction==1')
g2= pred.filter('prediction==2')
g3= pred.filter('prediction==3')
g4= pred.filter('prediction==4')
g5= pred.filter('prediction==5')
g6= pred.filter('prediction==6')
g7= pred.filter('prediction==7')
g8= pred.filter('prediction==8')
g9= pred.filter('prediction==9')
g10= pred.filter('prediction==10')
g11= pred.filter('prediction==11')
g12= pred.filter('prediction==12')
g13= pred.filter('prediction==13')
g14= pred.filter('prediction==14')
g15= pred.filter('prediction==15')
g16= pred.filter('prediction==16')
g17= pred.filter('prediction==17')
g18= pred.filter('prediction==18')
g19= pred.filter('prediction==19')


g0.write.parquet("/user/lsde22/label_0")
g1.write.parquet("/user/lsde22/label_1")
g2.write.parquet("/user/lsde22/label_2")
g3.write.parquet("/user/lsde22/label_3")
g4.write.parquet("/user/lsde22/label_4")
g5.write.parquet("/user/lsde22/label_5")
g6.write.parquet("/user/lsde22/label_6")
g7.write.parquet("/user/lsde22/label_7")
g8.write.parquet("/user/lsde22/label_8")
g9.write.parquet("/user/lsde22/label_9")
g10.write.parquet("/user/lsde22/label_10")
g11.write.parquet("/user/lsde22/label_11")
g12.write.parquet("/user/lsde22/label_12")
g13.write.parquet("/user/lsde22/label_13")
g14.write.parquet("/user/lsde22/label_14")
g15.write.parquet("/user/lsde22/label_15")
g16.write.parquet("/user/lsde22/label_16")
g17.write.parquet("/user/lsde22/label_17")
g18.write.parquet("/user/lsde22/label_18")
g19.write.parquet("/user/lsde22/label_19")
