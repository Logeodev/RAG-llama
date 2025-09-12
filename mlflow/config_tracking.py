import mlflow

mlflow.openai.autolog()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("RAG-llama")