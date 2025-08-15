from pyspark.sql.functions import *
from datetime import datetime, timedelta

class ComplianceMonitoring:
    """
    Module for ensuring regulatory compliance across operations
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def check_hours_of_service(self, driver_logs_df):
        """Check driver hours of service compliance"""
        window_spec = Window.partitionBy('driver_id').orderBy('date')
        
        return driver_logs_df.withColumn('prev_drive_end', lag('drive_end_time', 1).over(window_spec)) \
                           .withColumn('daily_drive_hours',
                                     (unix_timestamp(col('drive_end_time')) - 
                                      unix_timestamp(col('drive_start_time'))) / 3600) \
                           .withColumn('weekly_drive_hours',
                                     sum('daily_drive_hours').over(
                                         Window.partitionBy('driver_id')
                                               .orderBy('date')
                                               .rangeBetween(-6, 0)
                                     )) \
                           .withColumn('violation',
                                      when(col('daily_drive_hours') > 11, 'Daily limit exceeded')
                                      .when(col('weekly_drive_hours') > 60, 'Weekly limit exceeded')
                                      .when(
                                          (col('prev_drive_end').isNotNull()) &
                                          ((unix_timestamp(col('drive_start_time')) - 
                                            unix_timestamp(col('prev_drive_end'))) / 3600 < 10),
                                          'Rest period violation'
                                      ))
    
    def check_weight_limits(self, shipments_df):
        """Check vehicle weight limit compliance"""
        return shipments_df.withColumn('weight_violation',
                                     when(col('weight') > col('vehicle_max_weight'),
                                          concat('Over weight limit by ', 
                                                (col('weight') - col('vehicle_max_weight')).cast('int'),
                                                ' lbs'))
                                     .otherwise(None))
    
    def check_hazardous_materials(self, shipments_df, hazmat_regulations_df):
        """Check hazardous materials compliance"""
        return shipments_df.join(hazmat_regulations_df, 
                               shipments_df['hazmat_class'] == hazmat_regulations_df['class'], 
                               'left') \
                         .withColumn('hazmat_violation',
                                    when(
                                        (col('hazmat_class').isNotNull()) &
                                        (~col('vehicle_type').isin(col('allowed_vehicles'))),
                                        'Vehicle not approved for hazmat'
                                    ).when(
                                        (col('hazmat_class').isNotNull()) &
                                        (~col('route').isin(col('allowed_routes'))),
                                        'Route not approved for hazmat'
                                    ))
    
    def check_international_compliance(self, international_df, trade_regulations_df):
        """Check international shipping compliance"""
        return international_df.join(trade_regulations_df, 
                                   ['origin_country', 'destination_country'], 
                                   'left') \
                             .withColumn('export_violation',
                                        when(
                                            (col('export_license_required') == True) &
                                            (col('has_export_license') == False),
                                            'Missing export license'
                                        )) \
                             .withColumn('import_violation',
                                        when(
                                            (col('import_restrictions').isNotNull()) &
                                            (col('commodity_code').isin(col('restricted_commodities'))),
                                            'Import restriction violation'
                                        ))
    
    def generate_compliance_report(self, operations_data):
        """Generate comprehensive compliance report"""
        reports = {}
        
        # Hours of service
        if 'driver_logs' in operations_data:
            hos_report = self.check_hours_of_service(operations_data['driver_logs'])
            reports['hours_of_service'] = hos_report.filter(col('violation').isNotNull()) \
                                                  .groupBy('violation') \
                                                  .agg(count('*').alias('count')) \
                                                  .toPandas() \
                                                  .to_dict('records')
        
        # Weight limits
        if 'shipments' in operations_data:
            weight_report = self.check_weight_limits(operations_data['shipments'])
            reports['weight_limits'] = weight_report.filter(col('weight_violation').isNotNull()) \
                                                  .groupBy('weight_violation') \
                                                  .agg(count('*').alias('count')) \
                                                  .toPandas() \
                                                  .to_dict('records')
        
        # Hazardous materials
        if 'hazmat_shipments' in operations_data and 'hazmat_regulations' in operations_data:
            hazmat_report = self.check_hazardous_materials(
                operations_data['hazmat_shipments'],
                operations_data['hazmat_regulations']
            )
            reports['hazardous_materials'] = hazmat_report.filter(col('hazmat_violation').isNotNull()) \
                                                        .groupBy('hazmat_violation') \
                                                        .agg(count('*').alias('count')) \
                                                        .toPandas() \
                                                        .to_dict('records')
        
        # International trade
        if 'international_shipments' in operations_data and 'trade_regulations' in operations_data:
            trade_report = self.check_international_compliance(
                operations_data['international_shipments'],
                operations_data['trade_regulations']
            )
            reports['international_trade'] = {
                'export_violations': trade_report.filter(col('export_violation').isNotNull()) \
                                               .count(),
                'import_violations': trade_report.filter(col('import_violation').isNotNull()) \
                                               .count()
            }
        
        return reports