from pyspark.sql.functions import *
from pyspark.sql.window import Window

class ContractAnalysis:
    """
    Module for evaluating customer contract performance
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def analyze_contract_performance(self, shipments_df, contracts_df):
        """Analyze performance against contract terms"""
        return shipments_df.join(contracts_df, 'customer_id', 'inner') \
                         .groupBy('customer_id', 'contract_id') \
                         .agg(
                             count('*').alias('actual_shipments'),
                             first('min_shipments').alias('min_shipments'),
                             first('target_discount').alias('target_discount'),
                             avg('is_on_time').alias('on_time_rate'),
                             avg(when(col('service_level') == col('preferred_service'), 1)
                                .otherwise(0)).alias('preferred_service_usage'),
                             sum('revenue').alias('total_revenue')
                         ).withColumn('met_minimum',
                                    when(col('actual_shipments') >= col('min_shipments'), True)
                                    .otherwise(False))
    
    def calculate_discount_eligibility(self, performance_df):
        """Calculate discount eligibility based on contract terms"""
        return performance_df.withColumn('earned_discount',
                                       when(
                                           (col('met_minimum') & 
                                           (col('on_time_rate') >= 0.9) &
                                           (col('preferred_service_usage') >= 0.7),
                                           col('target_discount')
                                       .otherwise(0))
    
    def identify_underperforming_contracts(self, performance_df):
        """Identify underperforming contracts"""
        return performance_df.filter(
            (~col('met_minimum')) |
            (col('on_time_rate') < 0.8) |
            (col('preferred_service_usage') < 0.5)
        ).orderBy('total_revenue', ascending=False)
    
    def recommend_contract_changes(self, underperforming_df):
        """Recommend changes to underperforming contracts"""
        return underperforming_df.withColumn('recommendation',
                                           when(
                                               ~col('met_minimum'),
                                               'Renegotiate minimum shipment requirement'
                                           ).when(
                                               col('on_time_rate') < 0.8,
                                               'Adjust service level expectations'
                                           ).when(
                                               col('preferred_service_usage') < 0.5,
                                               'Revise pricing for preferred service'
                                           ).otherwise('General contract review needed'))
    
    def generate_contract_report(self, shipments_df, contracts_df):
        """Generate comprehensive contract analysis report"""
        # Analyze performance
        performance = self.analyze_contract_performance(shipments_df, contracts_df)
        
        # Calculate discounts
        discounts = self.calculate_discount_eligibility(performance)
        
        # Identify underperformers
        underperforming = self.identify_underperforming_contracts(performance)
        
        # Generate recommendations
        recommendations = self.recommend_contract_changes(underperforming)
        
        # Calculate totals
        total_contracts = contracts_df.count()
        contracts_meeting_goals = discounts.filter(col('earned_discount') > 0).count()
        total_discount_value = discounts.agg(sum('earned_discount')).collect()[0][0]
        
        return {
            'top_performing_contracts': discounts.orderBy('total_revenue', ascending=False)
                                               .limit(10)
                                               .toPandas()
                                               .to_dict('records'),
            'underperforming_contracts': recommendations.limit(10).toPandas().to_dict('records'),
            'total_contracts': total_contracts,
            'contracts_meeting_goals': contracts_meeting_goals,
            'total_discount_value': total_discount_value
        }