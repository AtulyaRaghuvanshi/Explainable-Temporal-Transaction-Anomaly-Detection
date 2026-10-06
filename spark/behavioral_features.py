"""Spark job: leakage-safe temporal behavioural features, computed on HDFS data.

Reads the raw IEEE-CIS CSVs from HDFS, joins identity, builds the same features as
src/features.py (basic + historical) with Spark window functions, and writes
  --features-out : full feature table (Parquet on HDFS) consumed by the Python ML pipeline
  --slim-out     : slim analytics table (Parquet on HDFS) exposed to Hive as an external table
"""
import argparse
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

ENTITIES = ['card1', 'addr1', 'DeviceInfo', 'P_emaildomain', 'ProductCD']
PAIRS = [('card1', 'addr1'), ('card1', 'DeviceInfo'), ('card1', 'P_emaildomain')]
# (name, type) of the columns Hive sees; source columns that are missing become NULL
SLIM = [('TransactionID', 'bigint'), ('TransactionDT', 'bigint'), ('isFraud', 'int'), ('TransactionAmt', 'double'),
        ('ProductCD', 'string'), ('card1', 'bigint'), ('card4', 'string'), ('card6', 'string'),
        ('P_emaildomain', 'string'), ('DeviceType', 'string'), ('tx_hour', 'int'), ('tx_weekday', 'int'),
        ('card1_hist_count', 'double'), ('card1_amount_z', 'double')]

p = argparse.ArgumentParser()
p.add_argument('--transactions', required=True)
p.add_argument('--identity')
p.add_argument('--features-out', required=True)
p.add_argument('--slim-out', required=True)
p.add_argument('--sample-frac', type=float, default=1.0)
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()

spark = SparkSession.builder.appName('XTADBehavioralFeatures').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
opts = dict(header=True, inferSchema=True)
df = spark.read.csv(a.transactions, **opts)
if a.identity:
    df = df.join(spark.read.csv(a.identity, **opts), on='TransactionID', how='left')
if a.sample_frac < 1:
    df = df.sample(fraction=a.sample_frac, seed=a.seed)

# basic features (same definitions as src/features.basic); missing_count excludes the label
feat_cols = [c for c in df.columns if c != 'isFraud']
df = (df.withColumn('tx_hour', ((F.col('TransactionDT') / 3600).cast('long') % 24).cast('int'))
        .withColumn('tx_day', (F.col('TransactionDT') / 86400).cast('long').cast('int'))
        .withColumn('tx_weekday', ((F.col('TransactionDT') / 86400).cast('long') % 7).cast('int'))
        .withColumn('amount_log', F.log1p(F.greatest(F.col('TransactionAmt'), F.lit(0.0))))
        .withColumn('missing_count', sum(F.col(f'`{c}`').isNull().cast('int') for c in feat_cols)))

# historical features: computed on a narrow projection (keeps window partitions small),
# then joined back to the wide table by TransactionID
ents = [c for c in ENTITIES if c in df.columns]
base = df.select('TransactionID', 'TransactionDT', 'TransactionAmt', *ents)
order = [F.col('TransactionDT'), F.col('TransactionID')]
hist = []
for c in ents:
    w = Window.partitionBy(c).orderBy(*order)
    prior = w.rowsBetween(Window.unboundedPreceding, -1)
    z = (F.col('TransactionAmt') - F.avg('TransactionAmt').over(prior)) / (
        F.coalesce(F.stddev_samp('TransactionAmt').over(prior), F.lit(0.0)) + F.lit(1e-6))
    base = (base.withColumn(f'{c}_hist_count', F.count(F.lit(1)).over(prior).cast('float'))
                .withColumn(f'{c}_amount_z', F.when(z.isNotNull(), F.least(F.greatest(z, F.lit(-20.0)), F.lit(20.0))).cast('float'))
                .withColumn(f'{c}_seconds_since_prev',  # NULL for a first occurrence (greatest() would turn it into 0)
                            F.when(F.lag('TransactionDT').over(w).isNotNull(),
                                   F.greatest(F.col('TransactionDT') - F.lag('TransactionDT').over(w), F.lit(0))).cast('float')))
    hist += [f'{c}_hist_count', f'{c}_amount_z', f'{c}_seconds_since_prev']
for x, y in PAIRS:
    if x in df.columns and y in df.columns:
        key = F.concat_ws('|', F.coalesce(F.col(x).cast('string'), F.lit('__NA__')),
                          F.coalesce(F.col(y).cast('string'), F.lit('__NA__')))
        base = base.withColumn('_k', key).withColumn(
            f'{x}_{y}_hist_count', (F.row_number().over(Window.partitionBy('_k').orderBy(*order)) - 1).cast('float')).drop('_k')
        hist.append(f'{x}_{y}_hist_count')

feats = df.join(base.select('TransactionID', *hist), on='TransactionID', how='inner').orderBy(*order)
feats.write.mode('overwrite').parquet(a.features_out)

full = spark.read.parquet(a.features_out)
slim = full.select(*[(F.col(n).cast(t) if n in full.columns else F.lit(None).cast(t)).alias(n) for n, t in SLIM])
slim.write.mode('overwrite').parquet(a.slim_out)
print('rows written:', full.count())
spark.stop()
