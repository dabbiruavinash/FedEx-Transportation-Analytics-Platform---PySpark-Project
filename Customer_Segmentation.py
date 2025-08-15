from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator

class CustomerSegmentation:
    """
    Module for clustering customers based on shipping patterns and value
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_customer_metrics(self, shipments_df):
        """Calculate customer metrics for segmentation"""
        return shipments_df.groupBy('customer_id') \
                         .agg(
                             count('*').alias('shipment_count'),
                             sum('revenue').alias('total_revenue'),
                             avg('revenue').alias('avg_revenue'),
                             avg('weight').alias('avg_weight'),
                             countDistinct('service_level').alias('service_levels_used'),
                             avg(when(col('is_on_time') == 1, 1).otherwise(0)).alias('on_time_rate')
                         )
    
    def prepare_clustering_features(self, customers_df):
        """Prepare features for customer clustering"""
        # Normalize features
        assembler = VectorAssembler(
            inputCols=['shipment_count', 'total_revenue', 'avg_revenue', 
                      'avg_weight', 'service_levels_used', 'on_time_rate'],
            outputCol="raw_features"
        )
        
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        pipeline = Pipeline(stages=[assembler, scaler])
        model = pipeline.fit(customers_df)
        
        return model.transform(customers_df)
    
    def cluster_customers(self, features_df, k=4):
        """Cluster customers using K-means"""
        kmeans = KMeans(
            k=k,
            featuresCol="features",
            predictionCol="cluster",
            seed=42
        )
        
        model = kmeans.fit(features_df)
        clustered = model.transform(features_df)
        
        # Evaluate clustering
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(clustered)
        
        return model, clustered, silhouette
    
    def analyze_clusters(self, clustered_df):
        """Analyze characteristics of each cluster"""
        return clustered_df.groupBy('cluster') \
                          .agg(
                              avg('shipment_count').alias('avg_shipments'),
                              avg('total_revenue').alias('avg_revenue'),
                              avg('avg_weight').alias('avg_weight'),
                              avg('service_levels_used').alias('avg_service_levels'),
                              avg('on_time_rate').alias('avg_on_time_rate'),
                              count('*').alias('customer_count')
                          ).orderBy('avg_revenue', ascending=False)
    
    def generate_segmentation_report(self, shipments_df, k=4):
        """Generate comprehensive customer segmentation report"""
        # Calculate metrics
        customer_metrics = self.calculate_customer_metrics(shipments_df)
        
        # Prepare features
        features_df = self.prepare_clustering_features(customer_metrics)
        
        # Cluster customers
        model, clustered_df, silhouette = self.cluster_customers(features_df, k)
        
        # Analyze clusters
        cluster_analysis = self.analyze_clusters(clustered_df)
        
        return {
            'silhouette_score': silhouette,
            'cluster_analysis': cluster_analysis.toPandas().to_dict('records'),
            'customer_segments': clustered_df.select('customer_id', 'cluster').toPandas().to_dict('records')
        }