import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import joblib

# Initialize Spark session
spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()

# Load the data
data = pd.read_csv(r"C:\Users\user\Documents\Projects\Customer Chain\WA_Fn-UseC_-Telco-Customer-Churn.csv")
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Convert pandas DataFrame to Spark DataFrame
df = spark.createDataFrame(data)

# Identify numeric and categorical columns
numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                        "PaperlessBilling", "PaymentMethod"]

# Impute missing values
imputer = Imputer(inputCols=numeric_features, outputCols=[f"{c}_imputed" for c in numeric_features])
df = imputer.fit(df).transform(df)

# StringIndexer for categorical columns
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").fit(df) for column in categorical_features]

# OneHotEncoder for categorical columns
encoders = [OneHotEncoder(inputCol=column + "_index", outputCol=column + "_ohe") for column in categorical_features]

# VectorAssembler to combine all features
assembler = VectorAssembler(inputCols=[f"{c}_imputed" for c in numeric_features] + [c + "_ohe" for c in categorical_features], outputCol="features")

# StandardScaler to scale features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Random Forest model
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="ChurnIndex")

# Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, rf])

# Convert target variable to numerical
df = df.withColumn("ChurnIndex", df["Churn"].cast("double"))

# Split the data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_df)

# Evaluate the model
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="ChurnIndex", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.2f}")

# Save the model
model.write().overwrite().save("churn_rf_model")
