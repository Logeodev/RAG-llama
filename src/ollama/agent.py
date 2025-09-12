from smolagents import LiteLLMModel
import mlflow

AgentModel = LiteLLMModel(model_id="ollama/llama3.1:8b", api_base="http://localhost:11435")

def set_mlflow_tracking(experiment_name: str):
    mlflow.smolagents.autolog()
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

def disable_mlflow_tracking():
    mlflow.smolagents.autolog(disable=True)