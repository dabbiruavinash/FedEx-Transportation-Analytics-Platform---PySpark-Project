from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

class DynamicPricing:
    """
    Module for dynamic pricing based on demand and capacity
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_price_elasticity(self, shipments_df):
        """Calculate price elasticity of demand by route and service level"""
        return shipments_df.groupBy('origin_region', 'destination_region', 'service_level') \
                         .agg(
                             avg('price').alias('avg_price'),
                             count('*').alias('shipment_count'),
                             (count('*') / avg('price')).alias('elasticity')
                         ).orderBy('elasticity', ascending=False)
    
    def analyze_capacity_utilization(self, shipments_df, capacity_df):
        """Analyze capacity utilization by route and time period"""
        return shipments_df.groupBy('origin_region', 'destination_region',
                                 weekofyear('shipment_date').alias('week')) \
                         .agg(count('*').alias('shipment_count')) \
                         .join(capacity_df, ['origin_region', 'destination_region', 'week'], 'left') \
                         .withColumn('utilization', col('shipment_count') / col('capacity')) \
                         .orderBy('utilization', ascending=False)
    
    def train_pricing_model(self, shipments_df):
        """Train model to predict optimal pricing"""
        # Prepare features
        origin_indexer = StringIndexer(inputCol="origin_region", outputCol="origin_index")
        origin_encoder = OneHotEncoder(inputCol="origin_index", outputCol="origin_encoded")
        
        dest_indexer = StringIndexer(inputCol="destination_region", outputCol="dest_index")
        dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_encoded")
        
        service_indexer = StringIndexer(inputCol="service_level", outputCol="service_index")
        service_encoder = OneHotEncoder(inputCol="service_index", outputCol="service_encoded")
        
        assembler = VectorAssembler(
            inputCols=["origin_encoded", "dest_encoded", "service_encoded",
                      "distance_miles", "weight", "week_of_year", "utilization"],
            outputCol="features"
        )
        
        # Define model
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="price",
            maxIter=100
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[
            origin_indexer, origin_encoder,
            dest_indexer, dest_encoder,
            service_indexer, service_encoder,
            assembler, gbt
        ])
        
        model = pipeline.fit(shipments_df)
        return model
    
    def recommend_pricing(self, model, current_df, demand_forecast_df, capacity_df):
        """Recommend pricing based on current conditions and forecasts"""
        # Prepare input data
        input_df = current_df.join(demand_forecast_df, ['origin_region', 'destination_region', 'week_of_year'], 'left') \
                           .join(capacity_df, ['origin_region', 'destination_region', 'week_of_year'], 'left') \
                           .withColumn('utilization', col('forecasted_shipments') / col('capacity'))
        
        # Make predictions
        predictions = model.transform(input_df)
        
        # Apply pricing rules
        return predictions.withColumn('recommended_price',
                                    when(col('utilization') > 0.9, col('prediction') * 1.2)
                                    .when(col('utilization') < 0.5, col('prediction') * 0.9)
                                    .otherwise(col('prediction'))) \
                         .select('origin_region', 'destination_region', 'service_level',
                                'week_of_year', 'forecasted_shipments', 'capacity',
                                'utilization', 'current_price', 'recommended_price')
    
    def generate_pricing_report(self, shipments_df, capacity_df, demand_forecast_df):
        """Generate comprehensive pricing analysis report"""
        # Calculate elasticity
        elasticity = self.calculate_price_elasticity(shipments_df)
        
        # Analyze capacity
        capacity_utilization = self.analyze_capacity_utilization(shipments_df, capacity_df)
        
        # Train model
        model = self.train_pricing_model(shipments_df)
        
        # Get current week
        current_week = shipments_df.agg(max(weekofyear('shipment_date'))).collect()[0][0]
        
        # Get current shipments
        current_shipments = shipments_df.filter(weekofyear(col('shipment_date')) == current_week)
        
        # Recommend pricing
        pricing_recommendations = self.recommend_pricing(model, current_shipments, demand_forecast_df, capacity_df)
        
        return {
            'price_elasticity': elasticity.limit(10).toPandas().to_dict('records'),
            'capacity_utilization': capacity_utilization.limit(10).toPandas().to_dict('records'),
            'pricing_recommendations': pricing_recommendations.limit(10).toPandas().to_dict('records')
        }