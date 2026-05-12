"""
AWS Lambda Handler for Agentic Logistics API
Deploy via SAM or direct Lambda upload
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import FastAPI app
from api_server import app, logger

# Mangum for ASGI to Lambda conversion
from mangum import Mangum

# ===== LAMBDA HANDLER =====

def lambda_handler(event, context):
    """
    AWS Lambda handler for FastAPI application
    
    Event structure (from API Gateway v2):
    {
        "requestContext": {
            "http": {
                "method": "POST",
                "path": "/analyze"
            }
        },
        "body": "...",
        "headers": {...}
    }
    """
    
    # Log incoming request
    logger.info(
        "Lambda invoke",
        request_id=context.request_id,
        function_name=context.function_name,
        memory_limit_in_mb=context.memory_limit_in_mb,
    )
    
    # Create Mangum handler and process
    handler = Mangum(app, lifespan="off")
    
    try:
        response = handler(event, context)
        logger.info(
            "Lambda invocation successful",
            request_id=context.request_id,
            status_code=response.get("statusCode", 200),
        )
        return response
    
    except Exception as e:
        logger.error(
            "Lambda invocation failed",
            request_id=context.request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        
        return {
            "statusCode": 500,
            "body": '{"error": "Internal server error"}',
            "headers": {"Content-Type": "application/json"},
        }


# ===== LOCAL TESTING =====

if __name__ == "__main__":
    # Mock Lambda context for testing
    class MockContext:
        request_id = "test-request-id"
        function_name = "agentic-logistics-api"
        memory_limit_in_mb = 3008
    
    # Test event
    test_event = {
        "requestContext": {
            "http": {
                "method": "POST",
                "path": "/analyze"
            }
        },
        "body": '{"predicted_days": 8.5, "promised_days": 7, "distance_km": 450, "weight_g": 1200, "freight_value": 45.0}',
        "headers": {"content-type": "application/json"},
    }
    
    result = lambda_handler(test_event, MockContext())
    print(result)
