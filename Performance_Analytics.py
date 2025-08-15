from pyspark.sql.functions import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PerformanceAnalytics:
    """
    Module for analyzing FedEx transportation performance metrics
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_on_time_delivery_rate(self, df):
        """Calculate overall on-time delivery rate"""
        return df.agg(
            (sum(col('is_on_time')) / count('*')).alias('on_time_delivery_rate')
        ).collect()[0]['on_time_delivery_rate']
    
    def service_level_performance(self, df):
        """Calculate performance by service level"""
        return df.groupBy('service_level') \
                .agg(
                    avg('transit_time_hours').alias('avg_transit_time'),
                    avg('is_on_time').alias('on_time_rate'),
                    count('*').alias('total_shipments')
                ).orderBy('on_time_rate', ascending=False)
    
    def route_performance_analysis(self, df, top_n=10):
        """Analyze performance by route (origin-destination pairs)"""
        route_stats = df.groupBy('origin_city', 'destination_city') \
                      .agg(
                          count('*').alias('shipment_count'),
                          avg('transit_time_hours').alias('avg_transit_time'),
                          avg('is_on_time').alias('on_time_rate'),
                          avg('delivery_delay_hours').alias('avg_delay_hours')
                      ).orderBy('shipment_count', ascending=False)
        
        # Get top routes by volume
        top_routes = route_stats.limit(top_n)
        
        # Get worst performing routes (by on-time rate) with minimum shipment threshold
        min_shipments = df.count() * 0.001  # 0.1% of total shipments
        worst_routes = route_stats.filter(col('shipment_count') >= min_shipments) \
                                 .orderBy('on_time_rate')
        
        return top_routes, worst_routes
    
    def time_trend_analysis(self, df, time_period='month'):
        """Analyze performance trends over time"""
        if time_period == 'month':
            time_col = date_format(col('shipment_date'), 'yyyy-MM').alias('month')
        elif time_period == 'week':
            time_col = date_format(col('shipment_date'), 'yyyy-ww').alias('week')
        else:  # default to day
            time_col = date_format(col('shipment_date'), 'yyyy-MM-dd').alias('day')
        
        return df.groupBy(time_col) \
                .agg(
                    count('*').alias('shipment_volume'),
                    avg('transit_time_hours').alias('avg_transit_time'),
                    avg('is_on_time').alias('on_time_rate'),
                    avg('delivery_delay_hours').alias('avg_delay_hours')
                ).orderBy(time_col)
    
    def vehicle_utilization_analysis(self, df):
        """Analyze vehicle utilization metrics"""
        return df.groupBy('vehicle_type') \
                .agg(
                    avg('weight_utilization').alias('avg_weight_utilization'),
                    avg('volume_utilization').alias('avg_volume_utilization'),
                    count('*').alias('trips_count')
                ).orderBy('avg_weight_utilization', ascending=False)
    
    def weather_impact_analysis(self, df):
        """Analyze impact of weather conditions on delivery performance"""
        if 'weather_condition' not in df.columns:
            raise ValueError("Weather data not available in DataFrame")
            
        return df.groupBy('weather_condition') \
                .agg(
                    count('*').alias('shipment_count'),
                    avg('is_on_time').alias('on_time_rate'),
                    avg('delivery_delay_hours').alias('avg_delay_hours'),
                    avg('transit_time_hours').alias('avg_transit_time')
                ).orderBy('on_time_rate')
    
    def correlation_analysis(self, df, numeric_columns):
        """Calculate correlation between numeric features"""
        # Convert to Pandas for correlation matrix (for demo purposes)
        # In production, would use Spark's correlation function for large datasets
        pdf = df.select(numeric_columns).toPandas()
        return pdf.corr()
    
    def plot_correlation_matrix(self, corr_matrix):
        """Plot correlation matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        return plt
    
    def generate_performance_report(self, df):
        """Generate comprehensive performance report"""
        report = {}
        
        # Basic metrics
        report['on_time_delivery_rate'] = self.calculate_on_time_delivery_rate(df)
        
        # Service level performance
        report['service_level_performance'] = self.service_level_performance(df).toPandas().to_dict('records')
        
        # Route performance
        top_routes, worst_routes = self.route_performance_analysis(df)
        report['top_routes'] = top_routes.toPandas().to_dict('records')
        report['worst_routes'] = worst_routes.toPandas().to_dict('records')
        
        # Monthly trends
        report['monthly_trends'] = self.time_trend_analysis(df, 'month').toPandas().to_dict('records')
        
        # Vehicle utilization
        report['vehicle_utilization'] = self.vehicle_utilization_analysis(df).toPandas().to_dict('records')
        
        # Weather impact (if data available)
        if 'weather_condition' in df.columns:
            report['weather_impact'] = self.weather_impact_analysis(df).toPandas().to_dict('records')
        
        return report