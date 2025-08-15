from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.sql.window import Window

class DemandForecasting:
    """
    Module for predicting shipment demand by region and time period
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def prepare_time_series_data(self, shipments_df):
        """Prepare time series data for demand forecasting"""
        return shipments_df.groupBy('origin_region', 'shipment_date') \
                         .agg(count('*').alias('daily_shipments'))
    
    def create_time_features(self, ts_df):
        """Create time-based features for forecasting"""
        return ts_df.withColumn('day_of_week', dayofweek(col('shipment_date'))) \
                  .withColumn('month', month(col('shipment_date'))) \
                  .withColumn('year', year(col('shipment_date'))) \
                  .withColumn('is_weekend', when(dayofweek(col('shipment_date')).isin([1, 7]), 1).otherwise(0))
    
    def add_lagged_features(self, ts_df, lag_days=[1, 7, 30]):
        """Add lagged features to time series data"""
        window_spec = Window.partitionBy('origin_region').orderBy('shipment_date')
        
        df_with_lags = ts_df
        for lag in lag_days:
            df_with_lags = df_with_lags.withColumn(f'shipments_lag_{lag}', 
                                                 lag('daily_shipments', lag).over(window_spec))
        
        return df_with_lags.na.fill(0)
    
    def train_demand_forecast_model(self, ts_df):
        """Train demand forecasting model"""
        # Prepare features
        region_indexer = StringIndexer(inputCol="origin_region", outputCol="region_index")
        region_encoder = OneHotEncoder(inputCol="region_index", outputCol="region_encoded")
        
        assembler = VectorAssembler(
            inputCols=["region_encoded", "day_of_week", "month", "year", "is_weekend",
                      "shipments_lag_1", "shipments_lag_7", "shipments_lag_30"],
            outputCol="features"
        )
        
        # Define model
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="daily_shipments",
            maxIter=100
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[region_indexer, region_encoder, assembler, gbt])
        model = pipeline.fit(ts_df)
        
        return model
    
    def forecast_demand(self, model, ts_df, forecast_days=30):
        """Generate demand forecasts for future periods"""
        # Get most recent data for each region
        window_spec = Window.partitionBy('origin_region').orderBy(col('shipment_date').desc())
        latest_data = ts_df.withColumn('rn', row_number().over(window_spec)) \
                          .filter(col('rn') == 1) \
                          .drop('rn')
        
        # Generate future dates
        max_date = ts_df.agg(max('shipment_date')).collect()[0][0]
        date_range = [max_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        # Create future records for each region
        regions = [row['origin_region'] for row in ts_df.select('origin_region').distinct().collect()]
        future_records = []
        
        for region in regions:
            for date in date_range:
                future_records.append({
                    'origin_region': region,
                    'shipment_date': date,
                    'daily_shipments': 0,  # Placeholder
                    'day_of_week': date.weekday() + 1,
                    'month': date.month,
                    'year': date.year,
                    'is_weekend': 1 if date.weekday() in [5, 6] else 0
                })
        
        future_df = self.spark.createDataFrame(future_records)
        
        # Add lagged features from historical data
        window_spec = Window.partitionBy('origin_region').orderBy('shipment_date')
        
        for lag in [1, 7, 30]:
            future_df = future_df.withColumn(f'shipments_lag_{lag}',
                                           lag('daily_shipments', lag).over(window_spec))
        
        # Fill NA lags with historical averages
        region_avg = ts_df.groupBy('origin_region') \
                         .agg(avg('daily_shipments').alias('region_avg'))
        
        future_df = future_df.join(broadcast(region_avg), 'origin_region', 'left')
        
        for lag in [1, 7, 30]:
            future_df = future_df.withColumn(f'shipments_lag_{lag}',
                                           coalesce(col(f'shipments_lag_{lag}'), col('region_avg')))
        
        # Make predictions
        return model.transform(future_df) \
                  .select('origin_region', 'shipment_date', 'prediction') \
                  .withColumnRenamed('prediction', 'forecasted_shipments') \
                  .orderBy('origin_region', 'shipment_date')