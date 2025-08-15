from pyspark.sql.functions import *
from pyspark.sql.window import Window

class PackageTracking:
    """
    Module for real-time package tracking and anomaly detection
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def process_tracking_events(self, tracking_df):
        """Process raw tracking events into shipment status timeline"""
        window_spec = Window.partitionBy('shipment_id').orderBy('event_timestamp')
        
        return tracking_df.withColumn('prev_status', lag('status', 1).over(window_spec)) \
                        .withColumn('prev_location', lag('location_id', 1).over(window_spec)) \
                        .withColumn('time_since_last_event',
                                  (unix_timestamp(col('event_timestamp')) - 
                                   unix_timestamp(lag('event_timestamp', 1).over(window_spec))) / 3600)
    
    def detect_anomalies(self, processed_df):
        """Detect anomalies in package tracking"""
        # Long delays between events
        delay_anomalies = processed_df.filter(col('time_since_last_event') > 24)
        
        # Unexpected location transitions
        location_rules = self.spark.read.csv('data/location_rules.csv', header=True)
        location_anomalies = processed_df.join(location_rules,
                                             (processed_df['prev_location'] == location_rules['from_location']) &
                                             (processed_df['location_id'] == location_rules['to_location']) &
                                             (processed_df['status'] == location_rules['status']),
                                             'left_anti')
        
        # Status regression (e.g., from "Out for delivery" back to "In transit")
        status_regression = processed_df.filter(
            ((col('status') == 'In transit') & (col('prev_status') == 'Out for delivery')) |
            ((col('status') == 'At facility') & (col('prev_status') == 'Out for delivery'))
        )
        
        return {
            'delays': delay_anomalies,
            'location_issues': location_anomalies,
            'status_regressions': status_regression
        }
    
    def calculate_estimated_delivery(self, tracking_df):
        """Calculate estimated delivery time based on current progress"""
        # Get most recent event for each shipment
        window_spec = Window.partitionBy('shipment_id').orderBy(col('event_timestamp').desc())
        latest_events = tracking_df.withColumn('rn', row_number().over(window_spec)) \
                                  .filter(col('rn') == 1) \
                                  .drop('rn')
        
        # Calculate estimated time remaining based on status
        return latest_events.withColumn('estimated_hours_remaining',
                                      when(col('status') == 'Delivered', 0)
                                      .when(col('status') == 'Out for delivery', 
                                           rand() * 4 + 1)  # 1-5 hours
                                      .when(col('status') == 'At facility',
                                           rand() * 12 + 12)  # 12-24 hours
                                      .otherwise(rand() * 48 + 24))  # 24-72 hours
    
    def generate_tracking_updates(self, tracking_df, customer_df):
        """Generate customer-facing tracking updates"""
        return tracking_df.join(customer_df, 'shipment_id', 'inner') \
                        .select('shipment_id', 'customer_id', 'status',
                               'location_id', 'event_timestamp',
                               'estimated_hours_remaining') \
                        .withColumn('customer_message',
                                   concat(
                                       lit('Your package is currently '),
                                       lower(col('status')),
                                       lit(' at '),
                                       col('location_id'),
                                       lit('. Estimated delivery in '),
                                       round(col('estimated_hours_remaining'), 1),
                                       lit(' hours.')
                                   ))
    
    def monitor_active_shipments(self, tracking_df):
        """Monitor and report on active shipments"""
        # Get latest status for each shipment
        window_spec = Window.partitionBy('shipment_id').orderBy(col('event_timestamp').desc())
        latest_status = tracking_df.withColumn('rn', row_number().over(window_spec)) \
                                 .filter(col('rn') == 1) \
                                 .drop('rn')
        
        # Categorize shipments
        on_time = latest_status.filter(
            (col('status') != 'Delivered') &
            (col('estimated_hours_remaining') <= 
             (col('promised_delivery_hours') - col('elapsed_hours')))
        )
        
        at_risk = latest_status.filter(
            (col('status') != 'Delivered') &
            (col('estimated_hours_remaining') > 
             (col('promised_delivery_hours') - col('elapsed_hours')))
        )
        
        delayed = latest_status.filter(
            (col('status') != 'Delivered') &
            (col('event_timestamp') > col('promised_delivery_time'))
        )
        
        return {
            'on_time': on_time,
            'at_risk': at_risk,
            'delayed': delayed
        }