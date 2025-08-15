from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
import numpy as np
from haversine import haversine

class NetworkOptimization:
    """
    Module for optimizing distribution network and hub locations
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def analyze_shipment_flows(self, shipments_df):
        """Analyze shipment flows between locations"""
        return shipments_df.groupBy('origin_region', 'destination_region') \
                         .agg(
                             count('*').alias('shipment_count'),
                             avg('distance_miles').alias('avg_distance'),
                             sum('weight').alias('total_weight')
                         ).orderBy('shipment_count', ascending=False)
    
    def identify_hub_locations(self, facilities_df, shipments_df, num_hubs=5):
        """Identify optimal hub locations using clustering"""
        # Prepare facility features
        assembler = VectorAssembler(
            inputCols=['facility_lat', 'facility_lon'],
            outputCol="features"
        )
        
        # K-means clustering
        kmeans = KMeans(
            k=num_hubs,
            featuresCol="features",
            predictionCol="hub_cluster",
            seed=42
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, kmeans])
        model = pipeline.fit(facilities_df)
        clustered = model.transform(facilities_df)
        
        # Calculate cluster centers
        centers = model.stages[-1].clusterCenters()
        hub_locations = [{'hub_id': i, 'lat': center[0], 'lon': center[1]} 
                        for i, center in enumerate(centers)]
        
        return clustered, self.spark.createDataFrame(hub_locations)
    
    def calculate_route_optimization(self, shipments_df, hubs_df):
        """Calculate potential savings from hub-and-spoke routing"""
        # Assign shipments to nearest hub
        shipments_with_hubs = shipments_df.crossJoin(hubs_df.hint("broadcast")) \
                                        .withColumn('distance_to_hub',
                                                  haversine(
                                                      col('origin_lat'), 
                                                      col('origin_lon'),
                                                      col('lat'),
                                                      col('lon')
                                                  )) \
                                        .withColumn('rn',
                                                  row_number().over(
                                                      Window.partitionBy('shipment_id')
                                                             .orderBy('distance_to_hub')
                                                  )) \
                                        .filter(col('rn') == 1) \
                                        .drop('rn')
        
        # Calculate current vs hub routing distances
        optimized = shipments_with_hubs.withColumn('current_distance', col('distance_miles')) \
                                     .withColumn('hub_distance',
                                                col('distance_to_hub') +
                                                haversine(
                                                    col('lat'), 
                                                    col('lon'),
                                                    col('destination_lat'),
                                                    col('destination_lon')
                                                ))
        
        # Calculate savings
        return optimized.withColumn('distance_savings',
                                  col('current_distance') - col('hub_distance')) \
                      .withColumn('cost_savings',
                                  col('distance_savings') * 0.15)  # $0.15 per mile savings
    
    def recommend_network_changes(self, savings_df, min_savings=100000):
        """Recommend network changes based on potential savings"""
        return savings_df.groupBy('hub_id', 'lat', 'lon') \
                       .agg(
                           sum('distance_savings').alias('total_distance_savings'),
                           sum('cost_savings').alias('total_cost_savings'),
                           count('*').alias('shipments_affected')
                       ).filter(col('total_cost_savings') >= min_savings) \
                       .orderBy('total_cost_savings', ascending=False)
    
    def generate_network_report(self, facilities_df, shipments_df):
        """Generate comprehensive network optimization report"""
        # Analyze flows
        flow_analysis = self.analyze_shipment_flows(shipments_df)
        
        # Identify hubs
        clustered_facilities, hub_locations = self.identify_hub_locations(facilities_df, shipments_df)
        
        # Calculate savings
        savings_df = self.calculate_route_optimization(shipments_df, hub_locations)
        
        # Generate recommendations
        recommendations = self.recommend_network_changes(savings_df)
        
        return {
            'top_flows': flow_analysis.limit(10).toPandas().to_dict('records'),
            'recommended_hubs': hub_locations.toPandas().to_dict('records'),
            'potential_savings': recommendations.toPandas().to_dict('records')
        }