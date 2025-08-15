#!/usr/bin/env python3
"""
FedEx Transportation Analytics Platform - Main Application
This is the core application that orchestrates all analytics modules.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from modules.data_ingestion import DataIngestion
from modules.data_processing import DataProcessing
from modules.data_quality_checks import DataQualityChecks
from modules.performance_analytics import PerformanceAnalytics
from modules.route_optimization import RouteOptimization
from modules.delivery_time_prediction import DeliveryTimePrediction
from modules.cost_optimization import CostOptimization
from modules.real_time_monitoring import RealTimeMonitoring
from modules.data_visualization import DataVisualization
from modules.vehicle_maintenance import VehicleMaintenanceAnalysis
from modules.driver_performance import DriverPerformanceAnalytics
from modules.fuel_efficiency import FuelEfficiencyAnalysis
from modules.inventory_management import InventoryManagement
from modules.demand_forecasting import DemandForecasting
from modules.customer_segmentation import CustomerSegmentation
from modules.revenue_analytics import RevenueAnalytics
from modules.package_tracking import PackageTracking
from modules.weather_impact import WeatherImpactModeling
from modules.dynamic_pricing import DynamicPricing
from modules.risk_assessment import RiskAssessment
from modules.compliance_monitoring import ComplianceMonitoring
from modules.network_optimization import NetworkOptimization
from modules.carbon_footprint import CarbonFootprintAnalysis
from modules.customer_satisfaction import CustomerSatisfactionAnalysis
from modules.incident_reporting import IncidentReporting
from modules.workforce_management import WorkforceManagement
from modules.insurance_analytics import InsuranceAnalytics
from modules.contract_analysis import ContractAnalysis
from modules.api_integration import APIIntegration
import yaml
import argparse
import logging
from datetime import datetime
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fedex_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FedExTransportationAnalytics:
    """
    Main application class for FedEx Transportation Analytics Platform
    """
    
    def __init__(self, config_path="config/app_config.yml"):
        """
        Initialize the FedEx Transportation Analytics application
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        logger.info("Initializing FedEx Transportation Analytics Platform")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Successfully loaded configuration from %s", config_path)
        except Exception as e:
            logger.error("Failed to load configuration: %s", str(e))
            raise
        
        # Initialize Spark
        try:
            self.ingestion = DataIngestion(config_path)
            self.spark = self.ingestion.spark
            logger.info("Spark session initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Spark session: %s", str(e))
            raise
        
        # Initialize all analytics modules
        try:
            logger.info("Initializing analytics modules...")
            
            # Core modules
            self.processing = DataProcessing(self.spark)
            self.quality = DataQualityChecks(self.spark)
            self.analytics = PerformanceAnalytics(self.spark)
            self.optimization = RouteOptimization(self.spark)
            self.prediction = DeliveryTimePrediction(self.spark)
            self.cost = CostOptimization(self.spark)
            self.monitoring = RealTimeMonitoring(self.spark)
            self.visualization = DataVisualization(self.spark)
            
            # Extended modules
            self.maintenance = VehicleMaintenanceAnalysis(self.spark)
            self.driver_analytics = DriverPerformanceAnalytics(self.spark)
            self.fuel = FuelEfficiencyAnalysis(self.spark)
            self.inventory = InventoryManagement(self.spark)
            self.demand = DemandForecasting(self.spark)
            self.customers = CustomerSegmentation(self.spark)
            self.revenue = RevenueAnalytics(self.spark)
            self.tracking = PackageTracking(self.spark)
            self.weather = WeatherImpactModeling(self.spark)
            self.pricing = DynamicPricing(self.spark)
            self.risk = RiskAssessment(self.spark)
            self.compliance = ComplianceMonitoring(self.spark)
            self.network = NetworkOptimization(self.spark)
            self.sustainability = CarbonFootprintAnalysis(self.spark)
            self.satisfaction = CustomerSatisfactionAnalysis(self.spark)
            self.incidents = IncidentReporting(self.spark)
            self.workforce = WorkforceManagement(self.spark)
            self.insurance = InsuranceAnalytics(self.spark)
            self.contracts = ContractAnalysis(self.spark)
            
            # API module
            self.api = APIIntegration(self.spark)
            
            logger.info("All analytics modules initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize analytics modules: %s", str(e))
            raise
    
    def _get_shipment_schema(self):
        """Define schema for shipment data"""
        return StructType([
            StructField("shipment_id", StringType(), False),
            StructField("customer_id", StringType(), True),
            StructField("service_level", StringType(), True),
            StructField("weight", DoubleType(), True),
            StructField("dimensions", StringType(), True),
            StructField("origin_address", StringType(), True),
            StructField("destination_address", StringType(), True),
            StructField("shipment_date", TimestampType(), True),
            StructField("delivery_date", TimestampType(), True),
            StructField("promised_delivery_date", TimestampType(), True),
            StructField("distance_miles", DoubleType(), True),
            StructField("vehicle_type", StringType(), True),
            StructField("vehicle_max_weight", DoubleType(), True),
            StructField("vehicle_max_volume", DoubleType(), True),
            StructField("length", DoubleType(), True),
            StructField("width", DoubleType(), True),
            StructField("height", DoubleType(), True),
            StructField("weather_condition", StringType(), True),
            StructField("destination_lat", DoubleType(), True),
            StructField("destination_lon", DoubleType(), True),
            StructField("declared_value", DoubleType(), True),
            StructField("fragile", BooleanType(), True),
            StructField("hazmat_class", StringType(), True),
            StructField("revenue", DoubleType(), True),
            StructField("cost", DoubleType(), True)
        ])
    
    def run_batch_analysis(self):
        """Run complete batch analysis pipeline"""
        logger.info("Starting FedEx Transportation Batch Analysis Pipeline")
        
        try:
            # 1. Data Ingestion
            logger.info("Step 1/10: Data Ingestion")
            shipment_schema = self._get_shipment_schema()
            shipments_df = self.ingestion.read_csv(
                self.config['data_paths']['shipments'],
                schema=shipment_schema
            )
            
            # 2. Data Quality Checks
            logger.info("Step 2/10: Data Quality Checks")
            dq_config = {
                'null_checks': ['shipment_id', 'customer_id', 'weight', 'shipment_date'],
                'value_ranges': {
                    'weight': (0, 1000),
                    'distance_miles': (0, 10000)
                },
                'duplicate_checks': [['shipment_id']],
                'expected_schema': shipment_schema,
                'delivery_consistency': True
            }
            dq_report = self.quality.generate_data_quality_report(shipments_df, dq_config)
            
            # 3. Data Processing
            logger.info("Step 3/10: Data Processing")
            processed_df = self.processing.apply_all_transformations(shipments_df)
            
            # 4. Performance Analytics
            logger.info("Step 4/10: Performance Analytics")
            performance_report = self.analytics.generate_performance_report(processed_df)
            
            # 5. Route Optimization
            logger.info("Step 5/10: Route Optimization")
            optimization_report = self.optimization.generate_optimization_report(processed_df)
            
            # 6. Delivery Time Prediction
            logger.info("Step 6/10: Delivery Time Prediction")
            train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
            prediction_report = self.prediction.generate_prediction_report(train_df, test_df)
            
            # 7. Cost Optimization
            logger.info("Step 7/10: Cost Optimization")
            cost_report = self.cost.generate_cost_optimization_report(processed_df)
            
            # 8. Extended Analytics
            logger.info("Step 8/10: Extended Analytics")
            maintenance_report = self.maintenance.generate_maintenance_schedule(
                self.ingestion.read_csv(self.config['data_paths']['vehicle_data'])
            )
            
            customer_report = self.customers.generate_segmentation_report(processed_df)
            
            # 9. Visualization
            logger.info("Step 9/10: Data Visualization")
            visualization_data = {
                'performance': performance_report,
                'optimization': optimization_report,
                'cost': cost_report,
                'customer_segmentation': customer_report
            }
            self.visualization.generate_comprehensive_report(
                visualization_data,
                self.config['output_paths']['visualizations']
            )
            
            # 10. Save Results
            logger.info("Step 10/10: Saving Results")
            processed_df.write.mode("overwrite").parquet(
                self.config['output_paths']['processed_data']
            )
            
            logger.info("Batch analysis completed successfully!")
            return {
                'data_quality': dq_report,
                'performance': performance_report,
                'optimization': optimization_report,
                'prediction': prediction_report,
                'cost': cost_report,
                'maintenance': maintenance_report,
                'customer': customer_report
            }
            
        except Exception as e:
            logger.error("Error during batch analysis: %s", str(e))
            raise
    
    def run_streaming_analysis(self):
        """Run real-time streaming analysis pipeline"""
        logger.info("Starting FedEx Transportation Streaming Analysis Pipeline")
        
        try:
            # 1. Initialize dashboard
            logger.info("Initializing real-time dashboard...")
            self.monitoring.create_real_time_monitoring_dashboard(
                self.config['data_paths']['dashboard_store']
            )
            
            # 2. Load reference data
            logger.info("Loading reference data...")
            reference_data = {
                'vehicles': self.ingestion.read_csv(self.config['data_paths']['vehicles']),
                'service_levels': self.ingestion.read_csv(self.config['data_paths']['service_levels'])
            }
            
            # 3. Load baseline stats
            logger.info("Loading baseline performance statistics...")
            baseline_stats = self.ingestion.read_csv(self.config['data_paths']['baseline_stats'])
            
            # 4. Start streaming processing
            logger.info("Starting streaming processing...")
            streaming_options = {
                'host': self.config['streaming']['host'],
                'port': self.config['streaming']['port'],
                'topic': self.config['streaming']['topic']
            }
            
            streaming_df = self.ingestion.read_stream(
                'kafka',
                {
                    'kafka.bootstrap.servers': f"{streaming_options['host']}:{streaming_options['port']}",
                    'subscribe': streaming_options['topic'],
                    'startingOffsets': 'latest'
                }
            )
            
            # 5. Process streaming data
            alerts_query = self.monitoring.process_real_time_events(
                streaming_df,
                reference_data,
                baseline_stats,
                self.config['data_paths']['dashboard_store']
            )
            
            # 6. Write alerts to console (in production would write to alerting system)
            alerts_query.writeStream \
                .outputMode("complete") \
                .format("console") \
                .option("truncate", "false") \
                .start()
            
            logger.info("Streaming analysis started. Waiting for termination...")
            alerts_query.awaitTermination()
            
        except Exception as e:
            logger.error("Error during streaming analysis: %s", str(e))
            raise
    
    def run_full_analysis(self):
        """Run complete end-to-end analytics pipeline"""
        logger.info("Starting Comprehensive FedEx Transportation Analysis")
        
        try:
            # 1. Data Ingestion and Preparation
            logger.info("Phase 1: Data Ingestion and Preparation")
            shipment_schema = self._get_shipment_schema()
            shipments_df = self.ingestion.read_csv(
                self.config['data_paths']['shipments'],
                schema=shipment_schema
            )
            
            # Data Quality Checks
            dq_report = self.quality.generate_data_quality_report(
                shipments_df,
                {
                    'null_checks': ['shipment_id', 'customer_id', 'weight', 'shipment_date'],
                    'value_ranges': {
                        'weight': (0, 1000),
                        'distance_miles': (0, 10000)
                    },
                    'duplicate_checks': [['shipment_id']],
                    'expected_schema': shipment_schema,
                    'delivery_consistency': True
                }
            )
            
            # Data Processing
            processed_df = self.processing.apply_all_transformations(shipments_df)
            
            # 2. Core Analytics
            logger.info("Phase 2: Core Analytics")
            performance_report = self.analytics.generate_performance_report(processed_df)
            optimization_report = self.optimization.generate_optimization_report(processed_df)
            cost_report = self.cost.generate_cost_optimization_report(processed_df)
            
            # 3. Extended Analytics
            logger.info("Phase 3: Extended Analytics")
            
            # Vehicle and Driver Analytics
            vehicle_data = self.ingestion.read_csv(self.config['data_paths']['vehicle_data'])
            maintenance_report = self.maintenance.generate_maintenance_schedule(vehicle_data)
            driver_report = self.driver_analytics.rank_drivers(
                self.ingestion.read_csv(self.config['data_paths']['driver_data'])
            )
            fuel_report = self.fuel.recommend_efficiency_improvements(vehicle_data)
            
            # Business Analytics
            customer_report = self.customers.generate_segmentation_report(processed_df)
            revenue_report = self.revenue.generate_revenue_report(
                processed_df,
                self.ingestion.read_csv(self.config['data_paths']['cost_data'])
            )
            demand_report = self.demand.forecast_demand(processed_df)
            
            # Risk and Compliance
            risk_report = self.risk.generate_risk_report(
                self.ingestion.read_csv(self.config['data_paths']['historical_incidents']),
                processed_df
            )
            compliance_report = self.compliance.generate_compliance_report({
                'driver_logs': self.ingestion.read_csv(self.config['data_paths']['driver_logs']),
                'shipments': processed_df,
                'hazmat_shipments': self.ingestion.read_csv(self.config['data_paths']['hazmat_shipments']),
                'hazmat_regulations': self.ingestion.read_csv(self.config['data_paths']['hazmat_regulations']),
                'international_shipments': self.ingestion.read_csv(self.config['data_paths']['international_shipments']),
                'trade_regulations': self.ingestion.read_csv(self.config['data_paths']['trade_regulations'])
            })
            
            # 4. Generate Outputs
            logger.info("Phase 4: Generating Outputs")
            
            # Visualization
            self.visualization.generate_comprehensive_report({
                'performance': performance_report,
                'optimization': optimization_report,
                'cost': cost_report,
                'customer': customer_report,
                'revenue': revenue_report,
                'demand': demand_report,
                'risk': risk_report
            }, self.config['output_paths']['reports'])
            
            # Save processed data
            processed_df.write.mode("overwrite").parquet(
                self.config['output_paths']['processed_data']
            )
            
            logger.info("Comprehensive analysis completed successfully!")
            return {
                'performance': performance_report,
                'optimization': optimization_report,
                'cost': cost_report,
                'maintenance': maintenance_report,
                'driver': driver_report,
                'fuel': fuel_report,
                'customer': customer_report,
                'revenue': revenue_report,
                'demand': demand_report,
                'risk': risk_report,
                'compliance': compliance_report
            }
            
        except Exception as e:
            logger.error("Error during comprehensive analysis: %s", str(e))
            raise
    
    def run_api_server(self):
        """Run the analytics API server"""
        logger.info("Starting FedEx Transportation Analytics API Server")
        try:
            self.api.run_api(
                host=self.config['api']['host'],
                port=self.config['api']['port']
            )
        except Exception as e:
            logger.error("Error running API server: %s", str(e))
            raise
    
    def shutdown(self):
        """Shutdown application"""
        logger.info("Shutting down FedEx Transportation Analytics Platform")
        try:
            self.ingestion.stop_all_streams()
            self.ingestion.stop_spark_session()
            logger.info("Application shutdown complete")
        except Exception as e:
            logger.error("Error during shutdown: %s", str(e))
            raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FedEx Transportation Analytics Platform')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['batch', 'streaming', 'full', 'api'], 
                       help='Execution mode: batch, streaming, full, or api')
    parser.add_argument('--config', type=str, default='config/app_config.yml',
                       help='Path to configuration file')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize and run application
        app = FedExTransportationAnalytics(args.config)
        
        try:
            if args.mode == 'batch':
                app.run_batch_analysis()
            elif args.mode == 'streaming':
                app.run_streaming_analysis()
            elif args.mode == 'full':
                app.run_full_analysis()
            elif args.mode == 'api':
                app.run_api_server()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down...")
        finally:
            app.shutdown()
    except Exception as e:
        logger.error("Application error: %s", str(e))
        raise