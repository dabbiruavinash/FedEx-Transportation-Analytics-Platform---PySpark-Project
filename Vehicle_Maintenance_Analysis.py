from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.sql.window import Window

class VehicleMaintenanceAnalysis:
    """
    Module for predicting vehicle maintenance needs based on usage patterns
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_usage_metrics(self, vehicle_df):
        """Calculate vehicle usage metrics"""
        window_spec = Window.partitionBy('vehicle_id').orderBy('date')
        
        return vehicle_df.withColumn('miles_since_last_maintenance', 
                                   col('odometer') - lag('odometer', 1).over(window_spec)) \
                        .withColumn('days_since_last_maintenance',
                                   datediff(col('date'), lag('date', 1).over(window_spec))) \
                        .withColumn('avg_daily_mileage',
                                   col('miles_since_last_maintenance') / col('days_since_last_maintenance'))
    
    def predict_maintenance_needs(self, vehicle_df):
        """Predict when next maintenance will be needed"""
        # Prepare features
        feature_cols = [
            'current_mileage',
            'days_since_last_service',
            'avg_daily_mileage',
            'vehicle_age',
            'last_service_type'
        ]
        
        # Encode categorical features
        indexer = StringIndexer(inputCol="last_service_type", outputCol="service_type_index")
        encoder = OneHotEncoder(inputCol="service_type_index", outputCol="service_type_encoded")
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols[:-1] + ["service_type_encoded"],
            outputCol="features"
        )
        
        # Define model
        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol="miles_until_next_service",
            numTrees=100
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[indexer, encoder, assembler, rf])
        model = pipeline.fit(vehicle_df)
        
        # Make predictions
        return model.transform(vehicle_df)
    
    def generate_maintenance_schedule(self, predictions_df):
        """Generate recommended maintenance schedule"""
        return predictions_df.withColumn('recommended_service_date',
                                      date_add(col('date'), 
                                              (col('prediction') / col('avg_daily_mileage')).cast('int'))) \
                           .select('vehicle_id', 'vehicle_type', 'current_mileage',
                                  'last_service_date', 'recommended_service_date',
                                  'prediction', 'avg_daily_mileage')
    
    def identify_urgent_maintenance(self, schedule_df, threshold_days=7):
        """Identify vehicles needing urgent maintenance"""
        return schedule_df.filter(
            datediff(col('recommended_service_date'), current_date()) <= threshold_days
        ).orderBy('recommended_service_date')