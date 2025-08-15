from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

class RiskAssessment:
    """
    Module for identifying high-risk shipments and routes
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_risk_factors(self, shipments_df):
        """Calculate risk factors for shipments"""
        return shipments_df.withColumn('value_density', col('declared_value') / col('weight')) \
                         .withColumn('fragility_risk', 
                                   when(col('fragile'), 1).otherwise(0)) \
                         .withColumn('high_value_risk',
                                   when(col('declared_value') > 1000, 1).otherwise(0)) \
                         .withColumn('weather_risk',
                                   when(col('weather_condition').isin(['storm', 'snow']), 1)
                                   .when(col('weather_condition') == 'rain', 0.5)
                                   .otherwise(0))
    
    def train_risk_model(self, shipments_df):
        """Train model to predict shipment risk"""
        # Prepare features
        origin_indexer = StringIndexer(inputCol="origin_region", outputCol="origin_index")
        origin_encoder = OneHotEncoder(inputCol="origin_index", outputCol="origin_encoded")
        
        service_indexer = StringIndexer(inputCol="service_level", outputCol="service_index")
        service_encoder = OneHotEncoder(inputCol="service_index", outputCol="service_encoded")
        
        assembler = VectorAssembler(
            inputCols=["origin_encoded", "service_encoded", "distance_miles",
                      "value_density", "fragility_risk", "high_value_risk", "weather_risk"],
            outputCol="features"
        )
        
        # Define model
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="had_incident",
            numTrees=100
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[
            origin_indexer, origin_encoder,
            service_indexer, service_encoder,
            assembler, rf
        ])
        
        model = pipeline.fit(shipments_df)
        return model
    
    def assess_shipment_risk(self, model, shipments_df):
        """Assess risk for new shipments"""
        # Calculate risk factors
        risk_factors = self.calculate_risk_factors(shipments_df)
        
        # Make predictions
        predictions = model.transform(risk_factors)
        
        return predictions.withColumn('risk_level',
                                    when(col('prediction') > 0.7, 'High')
                                    .when(col('prediction') > 0.4, 'Medium')
                                    .otherwise('Low'))
    
    def recommend_risk_mitigation(self, risk_df):
        """Recommend risk mitigation strategies"""
        return risk_df.withColumn('recommendation',
                                when(col('risk_level') == 'High',
                                     'Use premium service with additional insurance')
                                .when(col('risk_level') == 'Medium',
                                      'Add protective packaging and tracking')
                                .otherwise('Standard handling'))
    
    def generate_risk_report(self, historical_df, current_df):
        """Generate comprehensive risk assessment report"""
        # Train model
        model = self.train_risk_model(historical_df)
        
        # Assess risk for current shipments
        risk_assessment = self.assess_shipment_risk(model, current_df)
        
        # Generate recommendations
        recommendations = self.recommend_risk_mitigation(risk_assessment)
        
        # Aggregate risk by route
        route_risk = risk_assessment.groupBy('origin_region', 'destination_region') \
                                  .agg(
                                      avg('prediction').alias('avg_risk'),
                                      count('*').alias('shipment_count')
                                  ).orderBy('avg_risk', ascending=False)
        
        return {
            'high_risk_shipments': risk_assessment.filter(col('risk_level') == 'High')
                                                .limit(10)
                                                .toPandas()
                                                .to_dict('records'),
            'route_risk': route_risk.limit(10).toPandas().to_dict('records'),
            'recommendations_sample': recommendations.limit(10).toPandas().to_dict('records')
        }