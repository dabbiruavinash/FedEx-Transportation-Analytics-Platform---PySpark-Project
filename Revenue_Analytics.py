from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

class RevenueAnalytics:
    """
    Module for analyzing revenue streams and profitability
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_revenue_metrics(self, shipments_df):
        """Calculate revenue metrics by various dimensions"""
        # Revenue by service level
        service_revenue = shipments_df.groupBy('service_level') \
                                    .agg(
                                        sum('revenue').alias('total_revenue'),
                                        avg('revenue').alias('avg_revenue'),
                                        count('*').alias('shipment_count')
                                    ).orderBy('total_revenue', ascending=False)
        
        # Revenue by route
        route_revenue = shipments_df.groupBy('origin_region', 'destination_region') \
                                  .agg(
                                      sum('revenue').alias('total_revenue'),
                                      avg('revenue').alias('avg_revenue'),
                                      count('*').alias('shipment_count')
                                  ).orderBy('total_revenue', ascending=False)
        
        # Monthly revenue trends
        monthly_revenue = shipments_df.groupBy(year('shipment_date').alias('year'),
                                             month('shipment_date').alias('month')) \
                                    .agg(
                                        sum('revenue').alias('monthly_revenue'),
                                        count('*').alias('shipment_count')
                                    ).orderBy('year', 'month')
        
        return {
            'service_level': service_revenue,
            'route': route_revenue,
            'monthly': monthly_revenue
        }
    
    def analyze_profitability(self, shipments_df, cost_df):
        """Analyze shipment profitability"""
        return shipments_df.join(cost_df, 'shipment_id', 'inner') \
                         .withColumn('profit', col('revenue') - col('total_cost')) \
                         .withColumn('profit_margin', col('profit') / col('revenue'))
    
    def predict_revenue(self, shipments_df):
        """Predict revenue based on shipment characteristics"""
        # Prepare features
        service_indexer = StringIndexer(inputCol="service_level", outputCol="service_index")
        service_encoder = OneHotEncoder(inputCol="service_index", outputCol="service_encoded")
        
        origin_indexer = StringIndexer(inputCol="origin_region", outputCol="origin_index")
        origin_encoder = OneHotEncoder(inputCol="origin_index", outputCol="origin_encoded")
        
        dest_indexer = StringIndexer(inputCol="destination_region", outputCol="dest_index")
        dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_encoded")
        
        assembler = VectorAssembler(
            inputCols=["service_encoded", "origin_encoded", "dest_encoded", "weight", "distance_miles"],
            outputCol="features"
        )
        
        # Define model
        lr = LinearRegression(
            featuresCol="features",
            labelCol="revenue",
            maxIter=10
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[
            service_indexer, service_encoder,
            origin_indexer, origin_encoder,
            dest_indexer, dest_encoder,
            assembler, lr
        ])
        
        model = pipeline.fit(shipments_df)
        return model.transform(shipments_df)
    
    def identify_high_value_routes(self, profitability_df, threshold=0.3):
        """Identify high-value routes with good profit margins"""
        return profitability_df.groupBy('origin_region', 'destination_region') \
                             .agg(
                                 avg('profit_margin').alias('avg_profit_margin'),
                                 sum('revenue').alias('total_revenue'),
                                 count('*').alias('shipment_count')
                             ).filter(col('avg_profit_margin') >= threshold) \
                             .orderBy('total_revenue', ascending=False)
    
    def generate_revenue_report(self, shipments_df, cost_df):
        """Generate comprehensive revenue analysis report"""
        # Calculate metrics
        revenue_metrics = self.calculate_revenue_metrics(shipments_df)
        
        # Analyze profitability
        profitability = self.analyze_profitability(shipments_df, cost_df)
        
        # Identify high-value routes
        high_value_routes = self.identify_high_value_routes(profitability)
        
        return {
            'revenue_by_service': revenue_metrics['service_level'].toPandas().to_dict('records'),
            'revenue_by_route': revenue_metrics['route'].limit(10).toPandas().to_dict('records'),
            'monthly_trends': revenue_metrics['monthly'].toPandas().to_dict('records'),
            'high_value_routes': high_value_routes.limit(10).toPandas().to_dict('records'),
            'overall_profit_margin': profitability.agg(avg('profit_margin')).collect()[0][0]
        }