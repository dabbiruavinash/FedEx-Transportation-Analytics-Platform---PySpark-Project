from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from IPython.display import display

class DataVisualization:
    """
    Module for visualizing FedEx transportation data and analytics results
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def plot_service_level_performance(self, performance_df):
        """Plot performance metrics by service level"""
        pdf = performance_df.toPandas()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='service_level', y='on_time_rate', data=pdf)
        plt.title('On-Time Delivery Rate by Service Level')
        plt.ylabel('On-Time Rate')
        plt.xlabel('Service Level')
        plt.ylim(0, 1)
        plt.tight_layout()
        return plt
    
    def plot_route_performance(self, route_df, metric='on_time_rate', top_n=10):
        """Plot performance metrics for top routes"""
        pdf = route_df.orderBy(metric, ascending=False).limit(top_n).toPandas()
        pdf['route'] = pdf['origin_city'] + ' to ' + pdf['destination_city']
        
        plt.figure(figsize=(12, 8))
        sns.barplot(y='route', x=metric, data=pdf)
        plt.title(f'Top {top_n} Routes by {metric.replace("_", " ").title()}')
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel('Route')
        plt.tight_layout()
        return plt
    
    def plot_time_trends(self, trends_df, time_period='month'):
        """Plot performance trends over time"""
        pdf = trends_df.toPandas()
        
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Plot shipment volume
        ax1.plot(pdf[time_period], pdf['shipment_volume'], 'b-')
        ax1.set_xlabel(time_period.title())
        ax1.set_ylabel('Shipment Volume', color='b')
        ax1.tick_params('y', colors='b')
        
        # Plot on-time rate on second axis
        ax2 = ax1.twinx()
        ax2.plot(pdf[time_period], pdf['on_time_rate'], 'r-')
        ax2.set_ylabel('On-Time Rate', color='r')
        ax2.tick_params('y', colors='r')
        ax2.set_ylim(0, 1)
        
        plt.title('Shipment Volume and On-Time Rate Over Time')
        fig.tight_layout()
        return plt
    
    def plot_vehicle_utilization(self, utilization_df):
        """Plot vehicle utilization metrics"""
        pdf = utilization_df.toPandas()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='vehicle_type', y='avg_weight_utilization', data=pdf)
        plt.title('Average Weight Utilization by Vehicle Type')
        plt.ylabel('Utilization Rate')
        plt.xlabel('Vehicle Type')
        plt.ylim(0, 1)
        plt.tight_layout()
        return plt
    
    def plot_weather_impact(self, weather_df):
        """Plot impact of weather conditions on delivery performance"""
        pdf = weather_df.toPandas()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='weather_condition', y='on_time_rate', data=pdf)
        plt.title('On-Time Delivery Rate by Weather Condition')
        plt.ylabel('On-Time Rate')
        plt.xlabel('Weather Condition')
        plt.ylim(0, 1)
        plt.tight_layout()
        return plt
    
    def plot_feature_importance(self, importance_data, feature_names):
        """Plot feature importance from ML models"""
        indices, scores = zip(*importance_data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(scores), y=[feature_names[i] for i in indices])
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        return plt
    
    def plot_cost_analysis(self, cost_df):
        """Plot cost distribution and analysis"""
        pdf = cost_df.toPandas()
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='service_level', y='total_cost', data=pdf)
        plt.title('Cost Distribution by Service Level')
        plt.ylabel('Cost ($)')
        plt.xlabel('Service Level')
        plt.tight_layout()
        return plt
    
    def plot_optimized_routes(self, routes_df):
        """Plot optimized routes on a map (using Plotly)"""
        pdf = routes_df.toPandas()
        
        # Create map visualization
        fig = px.scatter_mapbox(pdf, 
                              lat="destination_lat", 
                              lon="destination_lon",
                              color="cluster_id",
                              hover_name="destination_address",
                              zoom=5)
        
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig
    
    def plot_real_time_metrics(self, metrics_df):
        """Plot real-time metrics dashboard"""
        pdf = metrics_df.toPandas()
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.lineplot(x='current_status_time', y='on_time_percentage', hue='service_level', data=pdf)
        plt.title('On-Time Percentage')
        plt.ylim(0, 1)
        
        plt.subplot(2, 2, 2)
        sns.lineplot(x='current_status_time', y='avg_delay', hue='service_level', data=pdf)
        plt.title('Average Delay (hours)')
        
        plt.subplot(2, 2, 3)
        sns.barplot(x='service_level', y='delayed_count', data=pdf)
        plt.title('Delayed Shipments Count')
        
        plt.subplot(2, 2, 4)
        sns.barplot(x='service_level', y='total_shipments', data=pdf)
        plt.title('Total Shipments')
        
        plt.tight_layout()
        return plt
    
    def generate_comprehensive_report(self, analysis_results, output_path=None):
        """Generate comprehensive visualization report"""
        figures = []
        
        # Service Level Performance
        if 'service_level_performance' in analysis_results:
            fig = self.plot_service_level_performance(analysis_results['service_level_performance'])
            figures.append(('Service Level Performance', fig))
        
        # Route Performance
        if 'route_performance' in analysis_results:
            fig = self.plot_route_performance(analysis_results['route_performance'])
            figures.append(('Top Routes Performance', fig))
        
        # Time Trends
        if 'time_trends' in analysis_results:
            fig = self.plot_time_trends(analysis_results['time_trends'])
            figures.append(('Time Trends', fig))
        
        # Vehicle Utilization
        if 'vehicle_utilization' in analysis_results:
            fig = self.plot_vehicle_utilization(analysis_results['vehicle_utilization'])
            figures.append(('Vehicle Utilization', fig))
        
        # Weather Impact
        if 'weather_impact' in analysis_results and analysis_results['weather_impact'] is not None:
            fig = self.plot_weather_impact(analysis_results['weather_impact'])
            figures.append(('Weather Impact', fig))
        
        # Cost Analysis
        if 'cost_analysis' in analysis_results:
            fig = self.plot_cost_analysis(analysis_results['cost_analysis']['detailed_stats']['service_levels'])
            figures.append(('Cost Analysis', fig))
        
        # Save or display figures
        if output_path:
            for name, fig in figures:
                fig.savefig(f"{output_path}/{name.replace(' ', '_')}.png")
                plt.close(fig)
            return True
        else:
            for name, fig in figures:
                display(fig)
                plt.close(fig)
            return True