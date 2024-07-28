from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
import pandas as pd

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()

# Load the model
model = PipelineModel.load("churn_rf_model")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    spark_df = spark.createDataFrame(df)

    # Transform the input data
    predictions = model.transform(spark_df)
    predicted_labels = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

    return jsonify(predicted_labels)

if __name__ == '__main__':
    app.run(debug=True)
