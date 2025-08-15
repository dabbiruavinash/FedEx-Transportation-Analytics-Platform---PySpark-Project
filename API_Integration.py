from flask import Flask, jsonify, request
from pyspark.sql import SparkSession
import json
from functools import wraps

class APIIntegration:
    """
    Module for providing REST API access to analytics results
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
        self.app = Flask(__name__)
        
        # Register routes
        self.register_routes()
    
    def spark_dataframe_to_json(self, df):
        """Convert Spark DataFrame to JSON"""
        return json.loads(df.toJSON().collect())
    
    def register_routes(self):
        """Register API routes"""
        
        # Health check endpoint
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy'})
        
        # Performance analytics endpoint
        @self.app.route('/api/performance', methods=['GET'])
        def get_performance():
            from modules.performance_analytics import PerformanceAnalytics
            analytics = PerformanceAnalytics(self.spark)
            shipments_df = self.spark.read.parquet('data/processed/shipments')
            report = analytics.generate_performance_report(shipments_df)
            return jsonify(report)
        
        # Route optimization endpoint
        @self.app.route('/api/optimize_routes', methods=['POST'])
        def optimize_routes():
            from modules.route_optimization import RouteOptimization
            optimizer = RouteOptimization(self.spark)
            shipments_df = self.spark.read.parquet('data/processed/shipments')
            report = optimizer.generate_optimization_report(shipments_df)
            return jsonify(report)
        
        # Delivery prediction endpoint
        @self.app.route('/api/predict_delivery', methods=['POST'])
        def predict_delivery():
            from modules.delivery_time_prediction import DeliveryTimePrediction
            predictor = DeliveryTimePrediction(self.spark)
            
            # Get data from request
            data = request.json
            shipments_df = self.spark.createDataFrame([data])
            
            # Make prediction
            result = predictor.predict_delivery_times(predictor.model, shipments_df)
            return jsonify(self.spark_dataframe_to_json(result))
    
    def run_api(self, host='0.0.0.0', port=5000):
        """Run the API server"""
        self.app.run(host=host, port=port)
    
    def create_api_client(self):
        """Create API client for internal use"""
        from flask.testing import FlaskClient
        return self.app.test_client()