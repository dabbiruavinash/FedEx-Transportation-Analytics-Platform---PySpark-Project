from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

class CustomerSatisfactionAnalysis:
    """
    Module for predicting and improving customer satisfaction
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def analyze_feedback(self, feedback_df):
        """Analyze customer feedback metrics"""
        return feedback_df.groupBy('service_level') \
                        .agg(
                            avg('rating').alias('avg_rating'),
                            avg('delivery_time_vs_expected').alias('avg_time_vs_expected'),
                            count(when(col('complaint') == True, 1)).alias('complaint_count'),
                            count('*').alias('feedback_count')
                        ).orderBy('avg_rating', ascending=False)
    
    def predict_satisfaction(self, shipments_df, feedback_df):
        """Predict customer satisfaction based on shipment characteristics"""
        # Join shipment data with feedback
        joined_df = shipments_df.join(feedback_df, 'shipment_id', 'inner')
        
        # Prepare features
        service_indexer = StringIndexer(inputCol="service_level", outputCol="service_index")
        service_encoder = OneHotEncoder(inputCol="service_index", outputCol="service_encoded")
        
        origin_indexer = StringIndexer(inputCol="origin_region", outputCol="origin_index")
        origin_encoder = OneHotEncoder(inputCol="origin_index", outputCol="origin_encoded")
        
        assembler = VectorAssembler(
            inputCols=["service_encoded", "origin_encoded", "distance_miles",
                      "delivery_delay_hours", "is_damaged", "contact_count"],
            outputCol="features"
        )
        
        # Define model
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="rating",
            maxIter=100
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[
            service_indexer, service_encoder,
            origin_indexer, origin_encoder,
            assembler, gbt
        ])
        
        model = pipeline.fit(joined_df)
        return model.transform(joined_df)
    
    def identify_dissatisfaction_factors(self, predictions_df):
        """Identify key factors contributing to customer dissatisfaction"""
        return predictions_df.filter(col('rating') < 3) \
                           .groupBy('service_level', 'origin_region') \
                           .agg(
                               avg('delivery_delay_hours').alias('avg_delay'),
                               avg('is_damaged').alias('damage_rate'),
                               avg('contact_count').alias('avg_contacts'),
                               count('*').alias('count')
                           ).orderBy('count', ascending=False)
    
    def recommend_improvements(self, analysis_df):
        """Recommend improvements to increase customer satisfaction"""
        return analysis_df.withColumn('recommendation',
                                    when(
                                        (col('avg_delay') > 24),
                                        'Improve on-time delivery for this route'
                                    ).when(
                                        (col('damage_rate') > 0.1),
                                        'Enhance package handling procedures'
                                    ).when(
                                        (col('avg_contacts') > 2),
                                        'Improve communication and tracking updates'
                                    ).otherwise('General service improvement needed'))
    
    def generate_satisfaction_report(self, shipments_df, feedback_df):
        """Generate comprehensive customer satisfaction report"""
        # Analyze feedback
        feedback_analysis = self.analyze_feedback(feedback_df)
        
        # Predict satisfaction
        predictions = self.predict_satisfaction(shipments_df, feedback_df)
        
        # Identify dissatisfaction factors
        dissatisfaction = self.identify_dissatisfaction_factors(predictions)
        
        # Generate recommendations
        recommendations = self.recommend_improvements(dissatisfaction)
        
        return {
            'feedback_analysis': feedback_analysis.toPandas().to_dict('records'),
            'dissatisfaction_factors': dissatisfaction.limit(10).toPandas().to_dict('records'),
            'improvement_recommendations': recommendations.limit(10).toPandas().to_dict('records'),
            'avg_rating': feedback_df.agg(avg('rating')).collect()[0][0],
            'complaint_rate': feedback_df.agg(avg(when(col('complaint') == True, 1).otherwise(0))).collect()[0][0]
        }