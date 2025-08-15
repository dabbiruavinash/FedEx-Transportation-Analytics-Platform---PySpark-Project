from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

class WeatherImpactModeling:
    """
    Module for advanced modeling of weather impact on operations
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def join_weather_data(self, shipments_df, weather_df):
        """Join shipment data with weather conditions"""
        return shipments_df.join(
            weather_df,
            (shipments_df['origin_city'] == weather_df['city']) &
            (shipments_df['shipment_date'] == weather_df['date']),
            'left'
        )
    
    def calculate_weather_impact(self, joined_df):
        """Calculate impact of weather on delivery performance"""
        return joined_df.groupBy('weather_condition') \
                      .agg(
                          avg('transit_time_hours').alias('avg_transit_time'),
                          avg('is_on_time').alias('on_time_rate'),
                          avg('delivery_delay_hours').alias('avg_delay'),
                          count('*').alias('shipment_count')
                      ).orderBy('on_time_rate')
    
    def predict_weather_impact(self, shipments_df, weather_df):
        """Predict impact of weather on new shipments"""
        # Join with weather data
        joined_df = self.join_weather_data(shipments_df, weather_df)
        
        # Prepare features
        weather_indexer = StringIndexer(inputCol="weather_condition", outputCol="weather_index")
        weather_encoder = OneHotEncoder(inputCol="weather_index", outputCol="weather_encoded")
        
        assembler = VectorAssembler(
            inputCols=["weather_encoded", "distance_miles", "service_level", "vehicle_type"],
            outputCol="features"
        )
        
        # Define model
        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol="transit_time_hours",
            numTrees=100
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[weather_indexer, weather_encoder, assembler, rf])
        model = pipeline.fit(joined_df)
        
        return model.transform(joined_df)
    
    def recommend_weather_contingencies(self, predictions_df):
        """Recommend contingency plans based on weather forecasts"""
        return predictions_df.withColumn('recommendation',
                                       when(
                                           (col('weather_condition').isin(['snow', 'storm'])) &
                                           (col('prediction') > col('transit_time_hours') * 1.5),
                                           'Delay shipment or upgrade service level'
                                       ).when(
                                           (col('weather_condition') == 'rain') &
                                           (col('prediction') > col('transit_time_hours') * 1.2),
                                           'Add weather protection to packages'
                                       ).otherwise('Proceed as normal'))
    
    def generate_weather_report(self, shipments_df, weather_df):
        """Generate comprehensive weather impact report"""
        # Join data
        joined_df = self.join_weather_data(shipments_df, weather_df)
        
        # Calculate impact
        impact_df = self.calculate_weather_impact(joined_df)
        
        # Predict impact
        predictions = self.predict_weather_impact(shipments_df, weather_df)
        
        # Generate recommendations
        recommendations = self.recommend_weather_contingencies(predictions)
        
        return {
            'weather_impact': impact_df.toPandas().to_dict('records'),
            'predictions_sample': predictions.limit(10).toPandas().to_dict('records'),
            'recommendations_sample': recommendations.limit(10).toPandas().to_dict('records')
        }