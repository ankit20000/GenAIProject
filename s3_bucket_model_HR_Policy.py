import json
import boto3
import os

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")

# Environment variables: set these in Lambda
BUCKET_NAME = os.environ["BUCKET_NAME"]      # e.g., labsaibucketdemo
FILE_KEY = os.environ["FILE_KEY"]            # e.g., HRPOLICY.txt
MODEL_ID = os.environ["MODEL_ID"]            # e.g., amazon.titan-text-express-v1

def get_context_from_s3():
    response = s3.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
    return response["Body"].read().decode("utf-8")

def lambda_handler(event, context):
    try:
        if "body" in event and isinstance(event["body"], str):
            body = json.loads(event["body"])
        else:
            body = event
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid request body", "details": str(e)})
        }

    user_input = body.get("question")
    if not user_input:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'question' in request body"})
        }

    # Load context from text file
    reference_text = get_context_from_s3()

    prompt = f"System: Use the following HR policy to answer the user's question.\nContext: {reference_text}\n\nHuman: {user_input}\nAssistant:"

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 500,
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
