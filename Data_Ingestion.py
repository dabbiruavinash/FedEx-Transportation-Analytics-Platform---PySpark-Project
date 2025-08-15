from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from configparser import ConfigParser
import yaml

class DataIngestion:
    """
    Module for ingesting various data sources related to FedEx transportation
    Handles CSV, JSON, Parquet, and database sources
    """
    
    def __init__(self, config_path="config/spark_config.yml"):
        self.config = self._load_config(config_path)
        self.spark = self._create_spark_session()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_spark_session(self):
        """Create and configure Spark session"""
        spark = SparkSession.builder \
            .appName("FedEx Transportation Analytics") \
            .master(self.config['spark']['master']) \
            .config("spark.executor.memory", self.config['spark']['executor_memory']) \
            .config("spark.driver.memory", self.config['spark']['driver_memory']) \
            .config("spark.sql.shuffle.partitions", self.config['spark']['shuffle_partitions']) \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel(self.config['spark']['log_level'])
        return spark
    
    def read_csv(self, file_path, schema=None, header=True, infer_schema=False):
        """Read CSV file with optional schema"""
        return self.spark.read \
            .option("header", header) \
            .option("inferSchema", infer_schema) \
            .csv(file_path, schema=schema)
    
    def read_json(self, file_path, multi_line=False):
        """Read JSON file"""
        return self.spark.read \
            .option("multiLine", multi_line) \
            .json(file_path)
    
    def read_parquet(self, file_path):
        """Read Parquet file"""
        return self.spark.read.parquet(file_path)
    
    def read_from_jdbc(self, jdbc_url, table, properties):
        """Read from JDBC source"""
        return self.spark.read \
            .jdbc(url=jdbc_url, table=table, properties=properties)
    
    def read_stream(self, source_type, options):
        """Read streaming data"""
        return self.spark.readStream \
            .format(source_type) \
            .options(**options) \
            .load()
    
    def validate_schema(self, df, expected_schema):
        """Validate DataFrame schema against expected schema"""
        actual_schema = set((field.name, str(field.dataType)) for field in df.schema.fields)
        expected = set((field.name, str(field.dataType)) for field in expected_schema.fields)
        
        if actual_schema != expected:
            missing = expected - actual_schema
            extra = actual_schema - expected
            raise ValueError(f"Schema mismatch. Missing: {missing}, Extra: {extra}")
        
        return True
    
    def write_to_sink(self, df, output_mode, format_type, path, checkpoint=None):
        """Write DataFrame to output sink"""
        writer = df.writeStream \
            .outputMode(output_mode) \
            .format(format_type)
            
        if checkpoint:
            writer = writer.option("checkpointLocation", checkpoint)
            
        return writer.start(path)
    
    def stop_all_streams(self):
        """Stop all active streaming queries"""
        for stream in self.spark.streams.active:
            stream.stop()
    
    def stop_spark_session(self):
        """Stop Spark session"""
        self.spark.stop()