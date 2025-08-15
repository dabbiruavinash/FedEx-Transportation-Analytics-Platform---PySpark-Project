from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np

class RouteOptimization:
    """
    Module for optimizing FedEx transportation routes using clustering and optimization algorithms
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def cluster_delivery_locations(self, df, num_clusters=10, features_col='features', prediction_col='cluster'):
        """
        Cluster delivery locations using K-means to identify delivery zones
        """
        # Prepare features (latitude and longitude)
        assembler = VectorAssembler(
            inputCols=['destination_lat', 'destination_lon'],
            outputCol=features_col
        )
        
        # K-means model
        kmeans = KMeans(
            k=num_clusters,
            featuresCol=features_col,
            predictionCol=prediction_col,
            seed=42
        )
        
        # Pipeline
        pipeline = Pipeline(stages=[assembler, kmeans])
        model = pipeline.fit(df)
        
        # Evaluate clustering by computing Silhouette score
        predictions = model.transform(df)
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        
        return model, predictions, silhouette
    
    def optimize_delivery_sequence(self, df, cluster_col='cluster'):
        """
        Optimize delivery sequence within each cluster using traveling salesman problem (TSP) approximation
        """
        # For each cluster, collect locations and calculate distance matrix
        # Note: This is a simplified approach - in production would use more sophisticated TSP solver
        
        # Get unique clusters
        clusters = df.select(cluster_col).distinct().rdd.flatMap(lambda x: x).collect()
        
        optimized_routes = []
        
        for cluster_id in clusters:
            # Get all locations in this cluster
            locations = df.filter(col(cluster_col) == cluster_id) \
                         .select('destination_lat', 'destination_lon', 'destination_address') \
                         .collect()
            
            if len(locations) > 1:
                # Calculate centroid as starting point
                centroid_lat = np.mean([loc['destination_lat'] for loc in locations])
                centroid_lon = np.mean([loc['destination_lon'] for loc in locations])
                
                # Sort locations by angle from centroid (simple TSP approximation)
                def angle_from_centroid(lat, lon):
                    return np.arctan2(lat - centroid_lat, lon - centroid_lon)
                
                sorted_locations = sorted(locations, 
                                        key=lambda x: angle_from_centroid(x['destination_lat'], x['destination_lon']))
                
                # Create route sequence
                for i, loc in enumerate(sorted_locations):
                    optimized_routes.append({
                        'cluster_id': cluster_id,
                        'sequence_num': i,
                        'destination_lat': loc['destination_lat'],
                        'destination_lon': loc['destination_lon'],
                        'destination_address': loc['destination_address']
                    })
            else:
                # Single location in cluster
                for loc in locations:
                    optimized_routes.append({
                        'cluster_id': cluster_id,
                        'sequence_num': 0,
                        'destination_lat': loc['destination_lat'],
                        'destination_lon': loc['destination_lon'],
                        'destination_address': loc['destination_address']
                    })
        
        # Convert to DataFrame
        return self.spark.createDataFrame(optimized_routes)
    
    def calculate_route_efficiency(self, original_df, optimized_df):
        """
        Calculate efficiency gains from route optimization
        """
        # In a real implementation, this would calculate actual distance savings
        # For this example, we'll just return some sample metrics
        
        # Count of clusters (delivery zones)
        num_zones = optimized_df.select('cluster_id').distinct().count()
        
        # Average locations per zone
        avg_locations_per_zone = optimized_df.count() / num_zones
        
        return {
            'delivery_zones_created': num_zones,
            'average_locations_per_zone': avg_locations_per_zone,
            'estimated_distance_reduction': '15-30% (simulated)',
            'estimated_time_savings': '20-35% (simulated)'
        }
    
    def recommend_vehicle_allocation(self, df, cluster_col='cluster'):
        """
        Recommend vehicle types for each delivery zone based on shipment characteristics
        """
        # Analyze shipment characteristics per cluster
        cluster_stats = df.groupBy(cluster_col) \
                         .agg(
                             avg('weight').alias('avg_weight'),
                             avg('volume_cubic_feet').alias('avg_volume'),
                             count('*').alias('shipment_count')
                         ).orderBy(cluster_col)
        
        # Define vehicle types and their capacities
        vehicle_types = [
            {'type': 'Small Van', 'max_weight': 2000, 'max_volume': 500, 'daily_capacity': 50},
            {'type': 'Medium Truck', 'max_weight': 10000, 'max_volume': 2000, 'daily_capacity': 30},
            {'type': 'Large Truck', 'max_weight': 25000, 'max_volume': 5000, 'daily_capacity': 20},
            {'type': 'Trailer', 'max_weight': 50000, 'max_volume': 10000, 'daily_capacity': 10}
        ]
        
        # Recommend vehicle type for each cluster
        def recommend_vehicle(avg_weight, avg_volume, shipment_count):
            for vehicle in sorted(vehicle_types, key=lambda x: x['max_weight']):
                if (avg_weight <= vehicle['max_weight'] * 0.7 and  # Use 70% of capacity for buffer
                    avg_volume <= vehicle['max_volume'] * 0.7):
                    # Calculate number of vehicles needed
                    vehicles_needed = max(1, round(shipment_count / vehicle['daily_capacity']))
                    return vehicle['type'], vehicles_needed
            return 'Trailer', max(1, round(shipment_count / 10))  # Default to trailer
        
        # Apply recommendation to each cluster
        recommendations = []
        for row in cluster_stats.collect():
            vehicle_type, vehicles_needed = recommend_vehicle(
                row['avg_weight'], row['avg_volume'], row['shipment_count'])
            
            recommendations.append({
                'cluster_id': row[cluster_col],
                'avg_weight': row['avg_weight'],
                'avg_volume': row['avg_volume'],
                'shipment_count': row['shipment_count'],
                'recommended_vehicle': vehicle_type,
                'vehicles_needed': vehicles_needed
            })
        
        return self.spark.createDataFrame(recommendations)
    
    def generate_optimization_report(self, original_df, num_clusters=10):
        """
        Generate comprehensive route optimization report
        """
        # Step 1: Cluster delivery locations
        _, clustered_df, silhouette = self.cluster_delivery_locations(original_df, num_clusters)
        
        # Step 2: Optimize delivery sequence within clusters
        optimized_routes = self.optimize_delivery_sequence(clustered_df)
        
        # Step 3: Calculate efficiency gains
        efficiency_metrics = self.calculate_route_efficiency(original_df, optimized_routes)
        
        # Step 4: Recommend vehicle allocation
        vehicle_recommendations = self.recommend_vehicle_allocation(clustered_df)
        
        return {
            'clustering_quality': {
                'silhouette_score': silhouette,
                'num_clusters': num_clusters
            },
            'efficiency_metrics': efficiency_metrics,
            'vehicle_recommendations': vehicle_recommendations.toPandas().to_dict('records'),
            'optimized_routes_sample': optimized_routes.limit(20).toPandas().to_dict('records')
        }