from pyspark.sql.functions import *
from pyspark.sql.window import Window

class DriverPerformanceAnalytics:
    """
    Module for analyzing driver performance and safety metrics
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_safety_metrics(self, trips_df):
        """Calculate driver safety metrics"""
        return trips_df.groupBy('driver_id') \
                     .agg(
                         avg('hard_brakes').alias('avg_hard_brakes'),
                         avg('hard_accels').alias('avg_hard_accels'),
                         avg('speeding_events').alias('avg_speeding_events'),
                         avg('mpg').alias('avg_mpg'),
                         count('*').alias('trips_count')
                     )
    
    def calculate_efficiency_metrics(self, trips_df):
        """Calculate driver efficiency metrics"""
        return trips_df.groupBy('driver_id') \
                     .agg(
                         avg('route_efficiency').alias('avg_route_efficiency'),
                         avg('idle_time').alias('avg_idle_time'),
                         avg('delivery_time_vs_estimate').alias('avg_time_vs_estimate')
                     )
    
    def rank_drivers(self, safety_df, efficiency_df):
        """Rank drivers by performance"""
        combined = safety_df.join(efficiency_df, 'driver_id', 'inner')
        
        # Calculate composite score (weights could be configurable)
        ranked = combined.withColumn('safety_score',
                                   (1 - (col('avg_hard_brakes') / 10) * 0.4 +
                                   (1 - (col('avg_hard_accels') / 10) * 0.3 +
                                   (1 - (col('avg_speeding_events') / 5) * 0.3)) \
                        .withColumn('efficiency_score',
                                   col('avg_route_efficiency') * 0.5 +
                                   (1 - (col('avg_idle_time') / 60) * 0.3 +
                                   (1 - (col('avg_time_vs_estimate') / 60) * 0.2)) \
                        .withColumn('composite_score',
                                   col('safety_score') * 0.6 + col('efficiency_score') * 0.4)
        
        # Add rank
        window_spec = Window.orderBy(col('composite_score').desc())
        return ranked.withColumn('rank', rank().over(window_spec))
    
    def identify_training_needs(self, ranked_df, safety_threshold=0.7, efficiency_threshold=0.7):
        """Identify drivers needing additional training"""
        return ranked_df.filter(
            (col('safety_score') < safety_threshold) |
            (col('efficiency_score') < efficiency_threshold)
        ).orderBy('composite_score')