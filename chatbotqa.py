import json
import boto3
import os

bedrock = boto3.client("bedrock-runtime")

def lambda_handler(event, context):
    try:
        if "body" in event and isinstance(event["body"], str):
            body = json.loads(event["body"])
        else:
            body = event
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON in request", "details": str(e)})
        }

    user_input = body.get("question")
    if not user_input:
        return {"statusCode": 400, "body": json.dumps({"error": "Question is required"})}

    model_id = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-text-express-v1")

    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "inputText": user_input,
            "textGenerationConfig": {
                "maxTokenCount": 200,
                "temperature": 0.5,
                "topP": 0.9
            }
        }),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    answer = result.get("results", [{}])[0].get("outputText", "No response")

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"answer": answer})
    }
