from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import datetime

class RealTimeMonitoring:
    """
    Module for real-time monitoring of FedEx transportation operations
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def process_streaming_data(self, streaming_df, reference_data):
        """Process real-time streaming data with reference data joins"""
        # Join with reference data (e.g., vehicle info, service levels)
        enriched_df = streaming_df.join(
            broadcast(reference_data['vehicles']),
            streaming_df['vehicle_id'] == reference_data['vehicles']['vehicle_id'],
            'left_outer'
        ).join(
            broadcast(reference_data['service_levels']),
            streaming_df['service_level_id'] == reference_data['service_levels']['level_id'],
            'left_outer'
        )
        
        # Calculate real-time metrics
        processed_df = enriched_df.withColumn('current_delay', 
                                           (unix_timestamp(col('current_status_time')) - 
                                            unix_timestamp(col('expected_status_time'))) / 3600) \
                                .withColumn('is_delayed', 
                                           when(col('current_delay') > 1, 1).otherwise(0))
        
        return processed_df
    
    def calculate_real_time_metrics(self, processed_df, time_window="5 minutes"):
        """Calculate real-time KPIs over sliding windows"""
        # Define window
        window_spec = Window.partitionBy('service_level').orderBy(col('current_status_time').cast("timestamp")) \
                          .rangeBetween(-windowDuration(time_window), windowDuration("0 seconds"))
        
        # Calculate metrics
        metrics_df = processed_df.withColumn('delayed_count', 
                                           sum(col('is_delayed')).over(window_spec)) \
                                .withColumn('total_shipments', 
                                           count('*').over(window_spec)) \
                                .withColumn('on_time_percentage', 
                                           (1 - (col('delayed_count') / col('total_shipments'))) \
                                .withColumn('avg_delay', 
                                           avg(col('current_delay')).over(window_spec))
        
        return metrics_df
    
    def detect_anomalies(self, metrics_df, baseline_stats):
        """Detect anomalies in real-time metrics compared to baselines"""
        # Compare current metrics to historical baselines
        anomalies_df = metrics_df.withColumn('is_anomaly',
                                           when(
                                               (col('on_time_percentage') < 
                                                col('baseline_on_time') - lit(0.1)) |  # 10% below baseline
                                               (col('avg_delay') > 
                                                col('baseline_avg_delay') + lit(0.5)),  # 0.5 hours above
                                               1
                                           ).otherwise(0))
        
        return anomalies_df.filter(col('is_anomaly') == 1)
    
    def generate_alerts(self, anomalies_df):
        """Generate alerts based on detected anomalies"""
        alerts_df = anomalies_df.select(
            'service_level',
            'current_status_time',
            'on_time_percentage',
            'baseline_on_time',
            'avg_delay',
            'baseline_avg_delay'
        ).withColumn('alert_message',
                   concat(
                       lit('Performance anomaly detected for '),
                       col('service_level'),
                       lit(' service: On-time '),
                       round((col('on_time_percentage') * 100), 1),
                       lit('% vs baseline '),
                       round((col('baseline_on_time') * 100), 1),
                       lit('%, Delay '),
                       round(col('avg_delay'), 1),
                       lit('h vs baseline '),
                       round(col('baseline_avg_delay'), 1),
                       lit('h')
                   ))
        
        return alerts_df
    
    def update_dashboard_metrics(self, metrics_df, dashboard_store_path):
        """Update dashboard metrics store with latest real-time metrics"""
        # Write metrics to delta lake for dashboard consumption
        metrics_df.write \
            .format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .save(dashboard_store_path)
        
        return True
    
    def process_real_time_events(self, streaming_df, reference_data, baseline_stats, dashboard_path):
        """Complete real-time event processing pipeline"""
        # Step 1: Enrich streaming data
        processed_df = self.process_streaming_data(streaming_df, reference_data)
        
        # Step 2: Calculate metrics
        metrics_df = self.calculate_real_time_metrics(processed_df)
        
        # Join with baseline stats
        metrics_with_baseline = metrics_df.join(
            broadcast(baseline_stats),
            'service_level',
            'left_outer'
        )
        
        # Step 3: Detect anomalies
        anomalies_df = self.detect_anomalies(metrics_with_baseline, baseline_stats)
        
        # Step 4: Generate alerts
        alerts_df = self.generate_alerts(anomalies_df)
        
        # Step 5: Update dashboard
        self.update_dashboard_metrics(metrics_with_baseline, dashboard_path)
        
        return alerts_df
    
    def create_real_time_monitoring_dashboard(self, dashboard_path):
        """Create initial dashboard structure for real-time monitoring"""
        # Create empty DataFrame with dashboard schema
        dashboard_schema = StructType([
            StructField("timestamp", TimestampType(), True),
            StructField("service_level", StringType(), True),
            StructField("on_time_percentage", DoubleType(), True),
            StructField("baseline_on_time", DoubleType(), True),
            StructField("avg_delay", DoubleType(), True),
            StructField("baseline_avg_delay", DoubleType(), True),
            StructField("delayed_count", IntegerType(), True),
            StructField("total_shipments", IntegerType(), True)
        ])
        
        empty_df = self.spark.createDataFrame([], dashboard_schema)
        
        # Write initial empty dashboard
        empty_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save(dashboard_path)
        
        return True