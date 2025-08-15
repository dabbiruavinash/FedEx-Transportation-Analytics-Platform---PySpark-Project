from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

class InsuranceAnalytics:
    """
    Module for analyzing insurance claims and risk factors
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def analyze_claim_patterns(self, claims_df):
        """Analyze insurance claim patterns"""
        return claims_df.groupBy('claim_type', 'root_cause') \
                      .agg(
                          count('*').alias('claim_count'),
                          avg('claim_amount').alias('avg_amount'),
                          sum('claim_amount').alias('total_amount')
                      ).orderBy('total_amount', ascending=False)
    
    def predict_claim_risk(self, shipments_df, historical_claims_df):
        """Predict claim risk for shipments"""
        # Join with historical claims
        joined_df = shipments_df.join(
            historical_claims_df.groupBy('route_id')
                             .agg(count('*').alias('historical_claims')),
            'route_id',
            'left'
        ).na.fill(0)
        
        # Prepare features
        service_indexer = StringIndexer(inputCol="service_level", outputCol="service_index")
        service_encoder = OneHotEncoder(inputCol="service_index", outputCol="service_encoded")
        
        assembler = VectorAssembler(
            inputCols=["service_encoded", "distance_miles", "historical_claims",
                      "value_density", "fragile"],
            outputCol="features"
        )
        
        # Define model
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="claim_amount",
            maxIter=100
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[service_indexer, service_encoder, assembler, gbt])
        model = pipeline.fit(historical_claims_df)
        
        return model.transform(joined_df)
    
    def recommend_risk_mitigation(self, risk_df):
        """Recommend risk mitigation strategies to reduce insurance costs"""
        return risk_df.withColumn('recommendation',
                                when(
                                    (col('prediction') > 1000),
                                    'Purchase additional insurance'
                                ).when(
                                    (col('prediction') > 500),
                                    'Add protective packaging and tracking'
                                ).otherwise('Standard coverage sufficient'))
    
    def calculate_premium_savings(self, recommendations_df):
        """Calculate potential premium savings from risk mitigation"""
        return recommendations_df.withColumn('potential_savings',
                                          when(
                                              col('recommendation') == 'Purchase additional insurance',
                                              col('prediction') * 0.1
                                          ).when(
                                              col('recommendation') == 'Add protective packaging and tracking',
                                              col('prediction') * 0.05
                                          ).otherwise(0)) \
                               .agg(sum('potential_savings')).collect()[0][0]
    
    def generate_insurance_report(self, claims_df, shipments_df):
        """Generate comprehensive insurance analysis report"""
        # Analyze claim patterns
        claim_analysis = self.analyze_claim_patterns(claims_df)
        
        # Predict risk
        risk_predictions = self.predict_claim_risk(shipments_df, claims_df)
        
        # Generate recommendations
        recommendations = self.recommend_risk_mitigation(risk_predictions)
        
        # Calculate savings
        potential_savings = self.calculate_premium_savings(recommendations)
        
        return {
            'claim_analysis': claim_analysis.limit(10).toPandas().to_dict('records'),
            'high_risk_shipments': risk_predictions.filter(col('prediction') > 500)
                                                 .limit(10)
                                                 .toPandas()
                                                 .to_dict('records'),
            'risk_mitigation_recommendations': recommendations.limit(10).toPandas().to_dict('records'),
            'total_claims': claims_df.count(),
            'total_claim_amount': claims_df.agg(sum('claim_amount')).collect()[0][0],
            'potential_savings': potential_savings
        }