from pyspark.sql.functions import *
from pyspark.sql.window import Window
import numpy as np

class CostOptimization:
    """
    Module for optimizing FedEx transportation costs
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_current_costs(self, df):
        """Calculate current operational costs based on shipment data"""
        # Define cost parameters (these would come from business rules in a real implementation)
        cost_params = {
            'base_cost': 5.0,  # Base cost per shipment
            'cost_per_mile': 0.15,  # Cost per mile
            'weight_surcharge': 0.02,  # Surcharge per pound over 10 lbs
            'express_surcharge': 10.0,  # Additional cost for express service
            'priority_surcharge': 5.0,  # Additional cost for priority service
            'weather_surcharge': {
                'clear': 0.0,
                'rain': 3.0,
                'snow': 7.0,
                'fog': 2.0,
                'storm': 10.0
            }
        }
        
        # Calculate costs for each shipment
        cost_df = df.withColumn('base_cost', lit(cost_params['base_cost'])) \
                   .withColumn('distance_cost', col('distance_miles') * lit(cost_params['cost_per_mile'])) \
                   .withColumn('weight_cost', 
                              when(col('weight') > 10, 
                                   (col('weight') - 10) * lit(cost_params['weight_surcharge']))
        
        # Add service level costs
        cost_df = cost_df.withColumn('service_cost',
                                   when(col('service_level') == 'Express', lit(cost_params['express_surcharge']))
                                   .when(col('service_level') == 'Priority', lit(cost_params['priority_surcharge']))
        
        # Add weather surcharges
        def get_weather_surcharge(weather):
            return cost_params['weather_surcharge'].get(weather.lower(), 0.0)
        
        weather_udf = udf(get_weather_surcharge, DoubleType())
        cost_df = cost_df.withColumn('weather_cost', weather_udf(col('weather_condition')))
        
        # Calculate total cost
        cost_df = cost_df.withColumn('total_cost', 
                                    col('base_cost') + col('distance_cost') + 
                                    col('weight_cost') + col('service_cost') + 
                                    col('weather_cost'))
        
        return cost_df
    
    def identify_cost_saving_opportunities(self, cost_df):
        """Identify potential cost saving opportunities"""
        # Analyze by service level
        service_level_stats = cost_df.groupBy('service_level') \
                                   .agg(
                                       avg('total_cost').alias('avg_cost'),
                                       count('*').alias('shipment_count'),
                                       sum('total_cost').alias('total_cost')
                                   ).orderBy('avg_cost', ascending=False)
        
        # Analyze by route
        route_stats = cost_df.groupBy('origin_city', 'destination_city') \
                           .agg(
                               avg('total_cost').alias('avg_cost'),
                               avg('distance_miles').alias('avg_distance'),
                               count('*').alias('shipment_count'),
                               sum('total_cost').alias('total_cost')
                           ).orderBy('total_cost', ascending=False)
        
        # Analyze by vehicle type
        vehicle_stats = cost_df.groupBy('vehicle_type') \
                              .agg(
                                  avg('total_cost').alias('avg_cost'),
                                  avg('weight_utilization').alias('avg_utilization'),
                                  count('*').alias('shipment_count'),
                                  sum('total_cost').alias('total_cost')
                              ).orderBy('avg_cost', ascending=False)
        
        # Find shipments where standard service could be used without significant delay
        potential_service_downgrades = cost_df.filter(
            (col('service_level').isin(['Express', 'Priority'])) &
            (col('delivery_delay_hours') < 2)  # Could tolerate 2 more hours delay
        )
        
        # Find underutilized vehicles
        underutilized_vehicles = cost_df.filter(
            (col('weight_utilization') < 0.5) |
            (col('volume_utilization') < 0.5)
        )
        
        return {
            'service_level_stats': service_level_stats,
            'route_stats': route_stats,
            'vehicle_stats': vehicle_stats,
            'potential_service_downgrades': potential_service_downgrades,
            'underutilized_vehicles': underutilized_vehicles
        }
    
    def simulate_cost_savings(self, cost_df, recommendations):
        """Simulate potential cost savings from recommendations"""
        # Calculate current total cost
        current_total_cost = cost_df.agg(sum('total_cost')).collect()[0][0]
        
        # Simulate service level downgrades
        if 'potential_service_downgrades' in recommendations:
            downgrades = recommendations['potential_service_downgrades']
            downgrade_count = downgrades.count()
            
            # Calculate savings from downgrading to standard service
            savings_df = downgrades.withColumn('potential_savings',
                                             col('service_cost') - lit(0.0))  # Standard service has no surcharge
            
            potential_savings = savings_df.agg(sum('potential_savings')).collect()[0][0]
        else:
            potential_savings = 0.0
            downgrade_count = 0
        
        # Simulate vehicle optimization
        if 'underutilized_vehicles' in recommendations:
            underutilized = recommendations['underutilized_vehicles']
            underutilized_count = underutilized.count()
            
            # Calculate potential savings from better vehicle utilization
            # (This is simplified - real calculation would be more complex)
            vehicle_savings = underutilized.agg(
                sum(col('total_cost') * lit(0.15))  # Estimate 15% savings
            ).collect()[0][0]
        else:
            vehicle_savings = 0.0
            underutilized_count = 0
        
        total_potential_savings = potential_savings + vehicle_savings
        savings_percentage = (total_potential_savings / current_total_cost) * 100
        
        return {
            'current_total_cost': current_total_cost,
            'potential_savings': total_potential_savings,
            'savings_percentage': savings_percentage,
            'service_downgrade_opportunities': downgrade_count,
            'vehicle_optimization_opportunities': underutilized_count
        }
    
    def recommend_cost_saving_actions(self, cost_analysis):
        """Generate actionable recommendations for cost savings"""
        recommendations = []
        
        # Service level recommendations
        service_stats = cost_analysis['service_level_stats'].collect()
        for row in service_stats:
            if row['service_level'] in ['Express', 'Priority']:
                recommendations.append({
                    'type': 'service_level',
                    'action': f"Evaluate downgrading some {row['service_level']} shipments to standard service",
                    'potential_impact': f"Average savings of ${row['avg_cost'] - service_stats[-1]['avg_cost']:.2f} per shipment",
                    'affected_shipments': row['shipment_count']
                })
        
        # Vehicle optimization recommendations
        vehicle_stats = cost_analysis['vehicle_stats'].collect()
        for row in vehicle_stats:
            if row['avg_utilization'] < 0.6:
                recommendations.append({
                    'type': 'vehicle_utilization',
                    'action': f"Improve utilization of {row['vehicle_type']} vehicles (current: {row['avg_utilization']*100:.1f}%)",
                    'potential_impact': "Estimated 10-20% cost reduction for underutilized vehicles",
                    'affected_shipments': row['shipment_count']
                })
        
        # Route optimization recommendations
        route_stats = cost_analysis['route_stats'].limit(5).collect()
        for row in route_stats:
            recommendations.append({
                'type': 'route_optimization',
                'action': f"Optimize route between {row['origin_city']} and {row['destination_city']}",
                'potential_impact': f"High-cost route (${row['avg_cost']:.2f} avg, ${row['total_cost']:.2f} total)",
                'affected_shipments': row['shipment_count']
            })
        
        return recommendations
    
    def generate_cost_optimization_report(self, df):
        """Generate comprehensive cost optimization report"""
        # Calculate current costs
        cost_df = self.calculate_current_costs(df)
        
        # Identify opportunities
        opportunities = self.identify_cost_saving_opportunities(cost_df)
        
        # Simulate savings
        savings_simulation = self.simulate_cost_savings(cost_df, opportunities)
        
        # Generate recommendations
        recommendations = self.recommend_cost_saving_actions(opportunities)
        
        # Prepare detailed stats for report
        service_stats = opportunities['service_level_stats'].toPandas().to_dict('records')
        route_stats = opportunities['route_stats'].limit(10).toPandas().to_dict('records')
        vehicle_stats = opportunities['vehicle_stats'].toPandas().to_dict('records')
        
        return {
            'current_cost_analysis': {
                'total_cost': savings_simulation['current_total_cost'],
                'avg_cost_per_shipment': cost_df.agg(avg('total_cost')).collect()[0][0],
                'cost_breakdown': {
                    'base': cost_df.agg(sum('base_cost')).collect()[0][0],
                    'distance': cost_df.agg(sum('distance_cost')).collect()[0][0],
                    'weight': cost_df.agg(sum('weight_cost')).collect()[0][0],
                    'service': cost_df.agg(sum('service_cost')).collect()[0][0],
                    'weather': cost_df.agg(sum('weather_cost')).collect()[0][0]
                }
            },
            'savings_opportunities': savings_simulation,
            'recommendations': recommendations,
            'detailed_stats': {
                'service_levels': service_stats,
                'routes': route_stats,
                'vehicles': vehicle_stats
            }
        }