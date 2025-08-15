from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

class InventoryManagement:
    """
    Module for tracking and predicting inventory levels at distribution centers
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_inventory_turnover(self, inventory_df):
        """Calculate inventory turnover metrics"""
        return inventory_df.groupBy('product_id', 'warehouse_id') \
                         .agg(
                             avg('daily_usage').alias('avg_daily_usage'),
                             stddev('daily_usage').alias('usage_stddev'),
                             count('*').alias('days_recorded')
                         )
    
    def predict_inventory_needs(self, inventory_df, lead_time_days=7):
        """Predict inventory needs based on historical usage"""
        # Prepare features
        window_spec = Window.partitionBy('product_id', 'warehouse_id').orderBy('date')
        
        features_df = inventory_df.withColumn('prev_usage', lag('daily_usage', 1).over(window_spec)) \
                                .withColumn('prev_usage_2', lag('daily_usage', 2).over(window_spec)) \
                                .withColumn('prev_usage_3', lag('daily_usage', 3).over(window_spec)) \
                                .na.fill(0)
        
        assembler = VectorAssembler(
            inputCols=["prev_usage", "prev_usage_2", "prev_usage_3", "day_of_week"],
            outputCol="features"
        )
        
        # Define model
        lr = LinearRegression(
            featuresCol="features",
            labelCol="daily_usage",
            maxIter=10
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, lr])
        model = pipeline.fit(features_df)
        
        # Make predictions
        predictions = model.transform(features_df)
        
        # Calculate safety stock
        return predictions.withColumn('safety_stock',
                                   col('prediction') * lead_time_days * 1.5)  # 1.5x buffer
    
    def generate_replenishment_orders(self, inventory_df, current_stock_df):
        """Generate replenishment orders based on inventory needs"""
        return inventory_df.join(current_stock_df, ['product_id', 'warehouse_id'], 'inner') \
                         .withColumn('order_quantity',
                                   when(col('current_quantity') < col('safety_stock'),
                                        col('safety_stock') - col('current_quantity'))
                                   .otherwise(0)) \
                         .filter(col('order_quantity') > 0) \
                         .select('product_id', 'warehouse_id', 'current_quantity',
                                'safety_stock', 'order_quantity')