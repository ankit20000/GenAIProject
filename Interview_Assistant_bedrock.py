import boto3
import json
import base64

s3 = boto3.client('s3')
textract = boto3.client('textract')
bedrock = boto3.client('bedrock-runtime')

BUCKET_NAME = "labsaibucketdemo"
PDF_KEY = "resumetest.pdf"
MODEL_ID = "amazon.titan-text-express-v1"

def extract_text_from_pdf(bucket, key):
    # Get PDF file from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    document_bytes = response['Body'].read()

    # Call Textract to detect document text
    textract_response = textract.detect_document_text(
        Document={'Bytes': document_bytes}
    )

    full_text = ""
    for item in textract_response['Blocks']:
        if item['BlockType'] == 'LINE':
            full_text += item['Text'] + '\n'

    return full_text.strip()

def lambda_handler(event, context):
    try:
        # Parse incoming event
        if "body" in event and isinstance(event["body"], str):
            body = json.loads(event["body"])
        else:
            body = event

        question = body.get("question")
        if not question:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'question' in request body"})
            }

        # Extract resume text from S3 PDF
        resume_text = extract_text_from_pdf(BUCKET_NAME, PDF_KEY)

        # Prepare input for Bedrock model
        prompt = f"System: Use the following resume to answer the question.\n\nResume:\n{resume_text}\n\nHuman: {question}\nAssistant:"

        # Call Amazon Bedrock Titan
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "inputText": prompt
            }),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        answer = result.get("results", [{}])[0].get("outputText", "No response found.")

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
