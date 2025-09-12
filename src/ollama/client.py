from openai import OpenAI
import mlflow

# https://mlflow.org/docs/latest/genai/tracing/integrations/listing/ollama/

OllamaClient = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key=""
)

def set_mlflow_tracking(experiment_name: str):
    mlflow.openai.autolog()
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

def disable_mlflow_tracking():
    mlflow.openai.autolog(disable=True)

def get_token_usage(trace_id = None):
    if trace_id is None:
        trace_id = mlflow.get_last_active_trace_id()

    trace = mlflow.get_trace(trace_id)

    total_usage = trace.info.token_usage
    print("== Total token usage: ==")
    print(f"  Input tokens: {total_usage['input_tokens']}")
    print(f"  Output tokens: {total_usage['output_tokens']}")
    print(f"  Total tokens: {total_usage['total_tokens']}")

    # Print the token usage for each LLM call
    print("\n== Detailed usage for each LLM call: ==")
    for span in trace.data.spans:
        if usage := span.get_attribute("mlflow.chat.tokenUsage"):
            print(f"{span.name}:")
            print(f"  Input tokens: {usage['input_tokens']}")
            print(f"  Output tokens: {usage['output_tokens']}")
            print(f"  Total tokens: {usage['total_tokens']}")