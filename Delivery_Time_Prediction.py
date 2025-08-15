from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    OneHotEncoder,
    StandardScaler
)
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, when

class DeliveryTimePrediction:
    """
    Module for predicting delivery times using machine learning models
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def prepare_features(self, df):
        """Prepare features for delivery time prediction"""
        # Select relevant columns
        feature_cols = [
            'distance_miles',
            'weight',
            'service_level',
            'origin_city',
            'destination_city',
            'shipment_day_of_week',
            'shipment_hour',
            'vehicle_type',
            'weather_condition'
        ]
        
        return df.select(feature_cols + ['transit_time_hours'])
    
    def preprocess_data(self, df):
        """Preprocess data for machine learning"""
        # Convert categorical columns to numeric
        categorical_cols = [
            'service_level',
            'origin_city',
            'destination_city',
            'vehicle_type',
            'weather_condition'
        ]
        
        # Create stages for each categorical column
        indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
            for col in categorical_cols
        ]
        
        encoders = [
            OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded")
            for col in categorical_cols
        ]
        
        # Assemble all features
        numeric_cols = ['distance_miles', 'weight', 'shipment_day_of_week', 'shipment_hour']
        encoded_cols = [f"{col}_encoded" for col in categorical_cols]
        
        assembler = VectorAssembler(
            inputCols=numeric_cols + encoded_cols,
            outputCol="features"
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=indexers + encoders + [assembler])
        model = pipeline.fit(df)
        processed_data = model.transform(df)
        
        return processed_data.select("features", col("transit_time_hours").alias("label"))
    
    def train_random_forest(self, train_df, test_df=None):
        """Train Random Forest regression model"""
        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=10,
            seed=42
        )
        
        # Train model
        model = rf.fit(train_df)
        
        # Evaluate on test data if provided
        if test_df:
            predictions = model.transform(test_df)
            evaluator = RegressionEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="rmse"
            )
            rmse = evaluator.evaluate(predictions)
            return model, rmse
        
        return model
    
    def train_gradient_boosting(self, train_df, test_df=None):
        """Train Gradient Boosting Trees regression model"""
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            maxDepth=5,
            seed=42
        )
        
        # Train model
        model = gbt.fit(train_df)
        
        # Evaluate on test data if provided
        if test_df:
            predictions = model.transform(test_df)
            evaluator = RegressionEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="rmse"
            )
            rmse = evaluator.evaluate(predictions)
            return model, rmse
        
        return model
    
    def hyperparameter_tuning(self, train_df, model_type='rf'):
        """Perform hyperparameter tuning using cross-validation"""
        if model_type == 'rf':
            # Define Random Forest model
            model = RandomForestRegressor(
                featuresCol="features",
                labelCol="label",
                seed=42
            )
            
            # Parameter grid
            param_grid = ParamGridBuilder() \
                .addGrid(model.numTrees, [50, 100, 150]) \
                .addGrid(model.maxDepth, [5, 10, 15]) \
                .build()
        else:
            # Define GBT model
            model = GBTRegressor(
                featuresCol="features",
                labelCol="label",
                seed=42
            )
            
            # Parameter grid
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxIter, [50, 100]) \
                .addGrid(model.maxDepth, [3, 5, 7]) \
                .build()
        
        # Evaluator
        evaluator = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="rmse"
        )
        
        # Cross-validator
        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            seed=42
        )
        
        # Run cross-validation
        cv_model = cv.fit(train_df)
        
        return cv_model
    
    def evaluate_model(self, model, test_df):
        """Evaluate model performance on test data"""
        predictions = model.transform(test_df)
        
        evaluator_rmse = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="rmse"
        )
        
        evaluator_r2 = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="r2"
        )
        
        evaluator_mae = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="mae"
        )
        
        rmse = evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        }
    
    def feature_importance(self, model):
        """Get feature importance from trained model"""
        if isinstance(model, RandomForestRegressor):
            # For Random Forest
            return list(zip(model.featureImportances.indices, model.featureImportances.values))
        elif isinstance(model, GBTRegressor):
            # For Gradient Boosting Trees
            return list(zip(model.featureImportances.indices, model.featureImportances.values))
        else:
            raise ValueError("Unsupported model type for feature importance")
    
    def predict_delivery_times(self, model, new_data):
        """Predict delivery times for new shipments"""
        # Prepare features
        processed_data = self.preprocess_data(new_data)
        
        # Make predictions
        predictions = model.transform(processed_data)
        
        # Add predictions to original DataFrame
        return new_data.withColumn("predicted_transit_time", predictions["prediction"])
    
    def generate_prediction_report(self, train_df, test_df, model_type='rf'):
        """Generate comprehensive prediction performance report"""
        # Preprocess data
        processed_train = self.preprocess_data(train_df)
        processed_test = self.preprocess_data(test_df)
        
        # Train model
        if model_type == 'rf':
            model, rmse = self.train_random_forest(processed_train, processed_test)
        else:
            model, rmse = self.train_gradient_boosting(processed_train, processed_test)
        
        # Evaluate model
        metrics = self.evaluate_model(model, processed_test)
        
        # Get feature importance
        importance = self.feature_importance(model)
        
        return {
            'model_type': model_type,
            'performance_metrics': metrics,
            'feature_importance': importance,
            'sample_predictions': model.transform(processed_test).limit(10).toPandas().to_dict('records')
        }