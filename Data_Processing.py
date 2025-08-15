from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from datetime import datetime, timedelta
import pytz

class DataProcessing:
    """
    Module for processing FedEx transportation data
    Includes cleaning, transformation, and feature engineering
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def clean_shipment_data(self, df):
        """Clean raw shipment data"""
        # Remove duplicates
        df = df.dropDuplicates(['shipment_id'])
        
        # Handle missing values
        df = df.fillna({
            'weight': 0,
            'dimensions': '0x0x0',
            'customer_id': 'UNKNOWN',
            'shipment_date': datetime.now(pytz.utc)
        })
        
        # Convert data types
        df = df.withColumn('weight', col('weight').cast(DoubleType())) \
               .withColumn('shipment_date', to_timestamp(col('shipment_date'))) \
               .withColumn('delivery_date', to_timestamp(col('delivery_date')))
        
        return df
    
    def process_geo_data(self, df):
        """Process geographical data"""
        # Extract city, state, and country from address
        df = df.withColumn('origin_city', split(col('origin_address'), ',')[0]) \
               .withColumn('origin_state', split(col('origin_address'), ',')[1]) \
               .withColumn('origin_country', split(col('origin_address'), ',')[2]) \
               .withColumn('destination_city', split(col('destination_address'), ',')[0]) \
               .withColumn('destination_state', split(col('destination_address'), ',')[1]) \
               .withColumn('destination_country', split(col('destination_address'), ',')[2])
        
        return df
    
    def calculate_transit_time(self, df):
        """Calculate transit time in hours"""
        return df.withColumn('transit_time_hours', 
                           (unix_timestamp(col('delivery_date')) - unix_timestamp(col('shipment_date'))) / 3600)
    
    def add_service_level_features(self, df):
        """Add features based on service level"""
        return df.withColumn('is_express', when(col('service_level').contains('Express'), 1).otherwise(0)) \
                 .withColumn('is_priority', when(col('service_level').contains('Priority'), 1).otherwise(0)) \
                 .withColumn('is_standard', when(col('service_level').contains('Standard'), 1).otherwise(0))
    
    def create_time_features(self, df):
        """Create time-based features"""
        return df.withColumn('shipment_year', year(col('shipment_date'))) \
                .withColumn('shipment_month', month(col('shipment_date'))) \
                .withColumn('shipment_day', dayofmonth(col('shipment_date'))) \
                .withColumn('shipment_day_of_week', dayofweek(col('shipment_date'))) \
                .withColumn('shipment_hour', hour(col('shipment_date')))
    
    def calculate_distance_bins(self, df):
        """Bin distances into categories"""
        return df.withColumn('distance_bin', 
                           when(col('distance_miles') < 100, 'Local')
                           .when((col('distance_miles') >= 100) & (col('distance_miles') < 500), 'Regional')
                           .when((col('distance_miles') >= 500) & (col('distance_miles') < 1000), 'National')
                           .otherwise('International'))
    
    def aggregate_shipments_by_route(self, df):
        """Aggregate shipments by origin-destination pairs"""
        window_spec = Window.partitionBy('origin_city', 'destination_city')
        
        return df.withColumn('total_shipments_route', count('shipment_id').over(window_spec)) \
                .withColumn('avg_weight_route', avg('weight').over(window_spec)) \
                .withColumn('avg_transit_time_route', avg('transit_time_hours').over(window_spec))
    
    def calculate_delivery_performance(self, df):
        """Calculate delivery performance metrics"""
        return df.withColumn('is_on_time', 
                           when(col('delivery_date') <= col('promised_delivery_date'), 1)
                           .otherwise(0)) \
                .withColumn('delivery_delay_hours',
                          (unix_timestamp(col('delivery_date')) - unix_timestamp(col('promised_delivery_date'))) / 3600)
    
    def create_vehicle_utilization_metrics(self, df):
        """Calculate vehicle utilization metrics"""
        return df.withColumn('volume_cubic_feet', 
                           col('length') * col('width') * col('height') / 1728) \
                .withColumn('weight_utilization', 
                          col('weight') / col('vehicle_max_weight')) \
                .withColumn('volume_utilization',
                          col('volume_cubic_feet') / col('vehicle_max_volume'))
    
    def join_with_weather_data(self, shipment_df, weather_df):
        """Join shipment data with weather data"""
        return shipment_df.join(weather_df,
                              (shipment_df['origin_city'] == weather_df['city']) &
                              (shipment_df['shipment_date'] == weather_df['date']),
                              'left_outer')
    
    def apply_all_transformations(self, df):
        """Apply all data processing transformations"""
        df = self.clean_shipment_data(df)
        df = self.process_geo_data(df)
        df = self.calculate_transit_time(df)
        df = self.add_service_level_features(df)
        df = self.create_time_features(df)
        df = self.calculate_distance_bins(df)
        df = self.aggregate_shipments_by_route(df)
        df = self.calculate_delivery_performance(df)
        return df