from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

class FuelEfficiencyAnalysis:
    """
    Module for analyzing and optimizing fuel consumption
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_fuel_metrics(self, trips_df):
        """Calculate fuel efficiency metrics"""
        return trips_df.withColumn('mpg', col('miles') / col('fuel_used')) \
                     .withColumn('fuel_cost', col('fuel_used') * col('fuel_price_per_gallon'))
    
    def analyze_fuel_factors(self, trips_df):
        """Analyze factors impacting fuel efficiency"""
        return trips_df.groupBy('vehicle_type', 'route_type', 'driver_id') \
                     .agg(
                         avg('mpg').alias('avg_mpg'),
                         stddev('mpg').alias('mpg_stddev'),
                         count('*').alias('trip_count')
                     ).orderBy('avg_mpg', ascending=False)
    
    def predict_fuel_consumption(self, trips_df):
        """Predict fuel consumption based on trip characteristics"""
        # Prepare features
        indexer = StringIndexer(inputCol="vehicle_type", outputCol="vehicle_type_index")
        encoder = OneHotEncoder(inputCol="vehicle_type_index", outputCol="vehicle_type_encoded")
        
        assembler = VectorAssembler(
            inputCols=["vehicle_type_encoded", "miles", "total_weight", "avg_speed", "route_elevation_change"],
            outputCol="features"
        )
        
        # Define model
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="fuel_used",
            maxIter=50
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[indexer, encoder, assembler, gbt])
        model = pipeline.fit(trips_df)
        
        return model.transform(trips_df)
    
    def recommend_efficiency_improvements(self, predictions_df):
        """Recommend fuel efficiency improvements"""
        return predictions_df.withColumn('potential_savings',
                                      (col('fuel_used') - col('prediction')) * col('fuel_price_per_gallon')) \
                           .select('trip_id', 'vehicle_id', 'driver_id',
                                  'fuel_used', 'prediction', 'potential_savings',
                                  'miles', 'total_weight') \
                           .orderBy('potential_savings', ascending=False)