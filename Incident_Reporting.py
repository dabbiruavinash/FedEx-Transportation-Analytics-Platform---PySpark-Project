from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

class IncidentReporting:
    """
    Module for tracking and analyzing transportation incidents
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def analyze_incident_trends(self, incidents_df):
        """Analyze incident trends by type and location"""
        return incidents_df.groupBy('incident_type', 'region') \
                         .agg(
                             count('*').alias('incident_count'),
                             avg('severity').alias('avg_severity'),
                             sum('cost').alias('total_cost')
                         ).orderBy('incident_count', ascending=False)
    
    def predict_incident_risk(self, shipments_df, historical_incidents_df):
        """Predict incident risk for shipments"""
        # Join with historical incidents by route
        joined_df = shipments_df.join(
            historical_incidents_df.groupBy('origin_region', 'destination_region')
                                 .agg(count('*').alias('historical_incidents')),
            ['origin_region', 'destination_region'],
            'left'
        ).na.fill(0)
        
        # Prepare features
        service_indexer = StringIndexer(inputCol="service_level", outputCol="service_index")
        service_encoder = OneHotEncoder(inputCol="service_index", outputCol="service_encoded")
        
        assembler = VectorAssembler(
            inputCols=["service_encoded", "distance_miles", "historical_incidents",
                      "weather_risk", "value_density"],
            outputCol="features"
        )
        
        # Define model (using historical data with incidents)
        training_data = historical_incidents_df.withColumn('had_incident', lit(1)) \
                                            .join(shipments_df, 'shipment_id', 'inner') \
                                            .union(
                                                shipments_df.join(
                                                    historical_incidents_df, 'shipment_id', 'left_anti'
                                                ).withColumn('had_incident', lit(0))
        
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="had_incident",
            numTrees=100
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[service_indexer, service_encoder, assembler, rf])
        model = pipeline.fit(training_data)
        
        return model.transform(joined_df)
    
    def recommend_incident_prevention(self, risk_df):
        """Recommend incident prevention measures"""
        return risk_df.withColumn('recommendation',
                                when(
                                    (col('prediction') > 0.7),
                                    'Assign experienced driver and add safety checks'
                                ).when(
                                    (col('prediction') > 0.4),
                                    'Add protective packaging and route optimization'
                                ).otherwise('Standard procedures'))
    
    def generate_incident_report(self, incidents_df, shipments_df):
        """Generate comprehensive incident analysis report"""
        # Analyze trends
        trend_analysis = self.analyze_incident_trends(incidents_df)
        
        # Predict risk
        risk_predictions = self.predict_incident_risk(shipments_df, incidents_df)
        
        # Generate recommendations
        recommendations = self.recommend_incident_prevention(risk_predictions)
        
        # Calculate key metrics
        total_incidents = incidents_df.count()
        avg_severity = incidents_df.agg(avg('severity')).collect()[0][0]
        total_cost = incidents_df.agg(sum('cost')).collect()[0][0]
        
        return {
            'incident_trends': trend_analysis.limit(10).toPandas().to_dict('records'),
            'high_risk_shipments': risk_predictions.filter(col('prediction') > 0.7)
                                                 .limit(10)
                                                 .toPandas()
                                                 .to_dict('records'),
            'prevention_recommendations': recommendations.limit(10).toPandas().to_dict('records'),
            'total_incidents': total_incidents,
            'avg_severity': avg_severity,
            'total_cost': total_cost
        }