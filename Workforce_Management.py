from pyspark.sql.functions import *
from pyspark.sql.window import Window

class WorkforceManagement:
    """
    Module for optimizing workforce scheduling and allocation
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def analyze_workload_distribution(self, assignments_df):
        """Analyze workload distribution across workers"""
        return assignments_df.groupBy('employee_id') \
                           .agg(
                               count('*').alias('assignment_count'),
                               sum('estimated_hours').alias('total_hours'),
                               avg('estimated_hours').alias('avg_hours_per_assignment')
                           ).orderBy('total_hours', ascending=False)
    
    def predict_workload(self, historical_df, forecast_df):
        """Predict workload requirements based on shipment forecasts"""
        # Calculate historical productivity
        productivity = historical_df.groupBy('facility_id', 'day_of_week') \
                                  .agg(
                                      (sum('units_processed') / sum('labor_hours')).alias('productivity'),
                                      avg('overtime_hours').alias('avg_overtime')
                                  )
        
        # Join with forecast and calculate required labor
        return forecast_df.join(productivity, ['facility_id', 'day_of_week'], 'left') \
                         .withColumn('required_labor_hours',
                                   col('forecasted_units') / coalesce(col('productivity'), lit(1))) \
                         .withColumn('recommended_staffing',
                                   ceil(col('required_labor_hours') / 8))  # 8-hour shifts
    
    def optimize_schedule(self, requirements_df, availability_df):
        """Optimize workforce schedule to meet requirements"""
        # Join requirements with employee availability
        joined_df = requirements_df.join(availability_df, ['facility_id', 'day_of_week'], 'inner') \
                                 .withColumn('preference_score',
                                           when(col('availability_type') == 'Preferred', 1)
                                           .when(col('availability_type') == 'Available', 0.5)
                                           .otherwise(0.1))
        
        # Assign shifts based on requirements and preferences
        window_spec = Window.partitionBy('facility_id', 'date').orderBy(col('preference_score').desc())
        
        return joined_df.withColumn('shift_assignment',
                                  when(
                                      row_number().over(window_spec) <= col('recommended_staffing'),
                                      'Assigned'
                                  ).otherwise('Not assigned'))
    
    def calculate_overtime_needs(self, schedule_df):
        """Calculate potential overtime needs"""
        return schedule_df.groupBy('employee_id') \
                        .agg(
                            sum(when(col('shift_assignment') == 'Assigned', 1).otherwise(0)).alias('assigned_shifts'),
                            sum('preference_score').alias('total_preference')
                        ).withColumn('overtime_likelihood',
                                   when(
                                       (col('assigned_shifts') > 5) & (col('total_preference') < 3),
                                       'High'
                                   ).when(
                                       (col('assigned_shifts') > 5),
                                       'Medium'
                                   ).otherwise('Low'))
    
    def generate_workforce_report(self, assignments_df, historical_df, forecast_df, availability_df):
        """Generate comprehensive workforce management report"""
        # Analyze current workload
        workload = self.analyze_workload_distribution(assignments_df)
        
        # Predict requirements
        requirements = self.predict_workload(historical_df, forecast_df)
        
        # Optimize schedule
        schedule = self.optimize_schedule(requirements, availability_df)
        
        # Analyze overtime
        overtime = self.calculate_overtime_needs(schedule)
        
        return {
            'workload_distribution': workload.limit(10).toPandas().to_dict('records'),
            'labor_requirements': requirements.groupBy('facility_id', 'date')
                                           .agg(
                                               first('recommended_staffing').alias('staff_needed'),
                                               sum('required_labor_hours').alias('total_hours_needed')
                                           )
                                           .limit(10)
                                           .toPandas()
                                           .to_dict('records'),
            'overtime_risk': overtime.filter(col('overtime_likelihood') != 'Low')
                                   .limit(10)
                                   .toPandas()
                                   .to_dict('records')
        }