from pyspark.sql.functions import *

class CarbonFootprintAnalysis:
    """
    Module for measuring and reducing environmental impact
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_emissions(self, shipments_df):
        """Calculate CO2 emissions for shipments"""
        # Emission factors (kg CO2 per ton-mile) by vehicle type
        emission_factors = {
            'Small Van': 0.25,
            'Medium Truck': 0.18,
            'Large Truck': 0.15,
            'Trailer': 0.12,
            'Air Freight': 0.60
        }
        
        # Create DataFrame with emission factors
        vehicle_types = self.spark.createDataFrame(
            [(k, v) for k, v in emission_factors.items()],
            ['vehicle_type', 'emission_factor']
        )
        
        # Calculate emissions
        return shipments_df.join(vehicle_types, 'vehicle_type', 'left') \
                         .withColumn('co2_emissions_kg',
                                   col('distance_miles') * (col('weight') / 2000) * 
                                   coalesce(col('emission_factor'), lit(0.2)))
    
    def analyze_emissions_by_route(self, emissions_df):
        """Analyze emissions by route"""
        return emissions_df.groupBy('origin_region', 'destination_region') \
                         .agg(
                             sum('co2_emissions_kg').alias('total_emissions'),
                             avg('co2_emissions_kg').alias('avg_emissions_per_shipment'),
                             count('*').alias('shipment_count')
                         ).orderBy('total_emissions', ascending=False)
    
    def identify_reduction_opportunities(self, emissions_df):
        """Identify opportunities to reduce emissions"""
        # High-emission routes
        high_emission_routes = self.analyze_emissions_by_route(emissions_df).limit(10)
        
        # Inefficient vehicles
        inefficient_vehicles = emissions_df.groupBy('vehicle_type') \
                                          .agg(
                                              avg('co2_emissions_kg').alias('avg_emissions'),
                                              count('*').alias('shipment_count')
                                          ).orderBy('avg_emissions', ascending=False)
        
        # Potential savings from mode shift
        mode_shift = emissions_df.withColumn('recommended_vehicle',
                                           when(
                                               (col('vehicle_type') == 'Air Freight') &
                                               (col('distance_miles') < 500),
                                               'Medium Truck'
                                           ).when(
                                               (col('vehicle_type').isin(['Large Truck', 'Trailer'])) &
                                               (col('weight') < 1000),
                                               'Medium Truck'
                                           ).otherwise(col('vehicle_type'))) \
                               .join(emissions_df.select('vehicle_type', 'emission_factor').distinct(),
                                     emissions_df['recommended_vehicle'] == emissions_df['vehicle_type'],
                                     'left') \
                               .withColumn('potential_savings',
                                          (col('emission_factor') - col('new_emission_factor')) * 
                                          col('distance_miles') * (col('weight') / 2000))
        
        return {
            'high_emission_routes': high_emission_routes,
            'inefficient_vehicles': inefficient_vehicles,
            'mode_shift_savings': mode_shift.agg(sum('potential_savings')).collect()[0][0]
        }
    
    def recommend_sustainability_measures(self, analysis_results):
        """Recommend sustainability measures"""
        recommendations = []
        
        # Route optimization
        if 'high_emission_routes' in analysis_results:
            for row in analysis_results['high_emission_routes'].collect():
                recommendations.append({
                    'type': 'route_optimization',
                    'action': f"Optimize route between {row['origin_region']} and {row['destination_region']}",
                    'potential_impact': f"Reduce emissions by ~{row['total_emissions'] * 0.15:.0f} kg CO2/year",
                    'priority': 'High'
                })
        
        # Vehicle upgrades
        if 'inefficient_vehicles' in analysis_results:
            for row in analysis_results['inefficient_vehicles'].limit(3).collect():
                recommendations.append({
                    'type': 'vehicle_upgrade',
                    'action': f"Replace {row['vehicle_type']} with more efficient models",
                    'potential_impact': f"Reduce emissions by ~{row['avg_emissions'] * row['shipment_count'] * 0.1:.0f} kg CO2/year",
                    'priority': 'Medium'
                })
        
        # Mode shift
        if 'mode_shift_savings' in analysis_results:
            recommendations.append({
                'type': 'mode_shift',
                'action': "Shift appropriate shipments to more efficient transportation modes",
                'potential_impact': f"Reduce emissions by ~{analysis_results['mode_shift_savings']:.0f} kg CO2/year",
                'priority': 'High'
            })
        
        return recommendations
    
    def generate_sustainability_report(self, shipments_df):
        """Generate comprehensive sustainability report"""
        # Calculate emissions
        emissions_df = self.calculate_emissions(shipments_df)
        
        # Analyze emissions
        emissions_analysis = self.analyze_emissions_by_route(emissions_df)
        
        # Identify opportunities
        opportunities = self.identify_reduction_opportunities(emissions_df)
        
        # Generate recommendations
        recommendations = self.recommend_sustainability_measures(opportunities)
        
        # Calculate totals
        total_emissions = emissions_df.agg(sum('co2_emissions_kg')).collect()[0][0]
        emissions_per_shipment = emissions_df.agg(avg('co2_emissions_kg')).collect()[0][0]
        
        return {
            'total_emissions_kg': total_emissions,
            'avg_emissions_per_shipment_kg': emissions_per_shipment,
            'high_emission_routes': opportunities['high_emission_routes'].limit(5).toPandas().to_dict('records'),
            'recommendations': recommendations
        }