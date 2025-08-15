from pyspark.sql.functions import col, count, when, isnull, sum as _sum
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

class DataQualityChecks:
    """
    Module for performing data quality checks on FedEx transportation data
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def check_for_nulls(self, df, columns):
        """Check for null values in specified columns"""
        null_counts = df.select([count(when(isnull(c), c)).alias(c) for c in columns]).collect()[0]
        return {col: null_counts[col] for col in columns}
    
    def validate_value_ranges(self, df, column_ranges):
        """
        Validate that column values fall within specified ranges
        column_ranges: dict of {'column_name': (min_value, max_value)}
        """
        results = {}
        for col_name, (min_val, max_val) in column_ranges.items():
            out_of_range = df.filter((col(col_name) < min_val) | (col(col_name) > max_val)).count()
            results[col_name] = out_of_range
        return results
    
    def check_for_duplicates(self, df, key_columns):
        """Check for duplicate records based on key columns"""
        total_count = df.count()
        distinct_count = df.dropDuplicates(key_columns).count()
        return total_count - distinct_count
    
    def validate_schema(self, df, expected_schema):
        """Validate DataFrame schema matches expected schema"""
        actual_schema = df.schema
        if actual_schema != expected_schema:
            missing = set(expected_schema) - set(actual_schema)
            extra = set(actual_schema) - set(expected_schema)
            return False, {"missing_columns": list(missing), "extra_columns": list(extra)}
        return True, {}
    
    def check_referential_integrity(self, fact_df, dimension_df, join_keys):
        """Check referential integrity between fact and dimension tables"""
        # Find records in fact table that don't exist in dimension table
        missing_records = fact_df.join(dimension_df, join_keys, "left_anti")
        return missing_records.count()
    
    def validate_date_consistency(self, df, date_columns):
        """Validate that dates are consistent (e.g., delivery date after shipment date)"""
        results = {}
        for col_name in date_columns:
            invalid_dates = df.filter(col(col_name).isNull() | (col(col_name) < lit("1970-01-01"))).count()
            results[col_name] = invalid_dates
        return results
    
    def check_delivery_consistency(self, df):
        """Check that delivery dates are after shipment dates"""
        return df.filter(col("delivery_date") < col("shipment_date")).count()
    
    def generate_data_quality_report(self, df, checks_config):
        """Generate comprehensive data quality report"""
        report = {}
        
        # Null checks
        if 'null_checks' in checks_config:
            report['null_counts'] = self.check_for_nulls(df, checks_config['null_checks'])
        
        # Value range checks
        if 'value_ranges' in checks_config:
            report['value_range_violations'] = self.validate_value_ranges(df, checks_config['value_ranges'])
        
        # Duplicate checks
        if 'duplicate_checks' in checks_config:
            report['duplicate_counts'] = {}
            for check in checks_config['duplicate_checks']:
                report['duplicate_counts'][str(check)] = self.check_for_duplicates(df, check)
        
        # Schema validation
        if 'expected_schema' in checks_config:
            is_valid, schema_issues = self.validate_schema(df, checks_config['expected_schema'])
            report['schema_validation'] = {
                'is_valid': is_valid,
                'issues': schema_issues
            }
        
        # Delivery consistency
        if 'delivery_consistency' in checks_config and checks_config['delivery_consistency']:
            report['delivery_consistency_issues'] = self.check_delivery_consistency(df)
        
        return report
    
    def create_quality_metrics_table(self, dq_report):
        """Create a DataFrame with data quality metrics"""
        metrics = []
        
        # Add null check metrics
        for col_name, null_count in dq_report.get('null_counts', {}).items():
            metrics.append(("null_check", col_name, null_count))
        
        # Add value range metrics
        for col_name, range_violations in dq_report.get('value_range_violations', {}).items():
            metrics.append(("value_range", col_name, range_violations))
        
        # Add duplicate metrics
        for key_cols, dup_count in dq_report.get('duplicate_counts', {}).items():
            metrics.append(("duplicate_check", str(key_cols), dup_count))
        
        # Create DataFrame
        schema = StructType([
            StructField("check_type", StringType(), False),
            StructField("check_parameter", StringType(), False),
            StructField("issue_count", IntegerType(), False)
        ])
        
        return self.spark.createDataFrame(metrics, schema)