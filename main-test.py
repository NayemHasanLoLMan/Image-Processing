import openai
import os
import json
import base64
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set API key for older OpenAI library
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_prescription_info(image_path=None, image_url=None):
    """
    Extract prescription information from an image
    Args:
        image_path: Local path to image file
        image_url: URL to image (use either image_path or image_url)
    Returns:
        JSON with prescription information
    """
    
    # Prepare image content
    if image_path:
        # Read and encode local image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    elif image_url:
        # Use image URL directly
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }
    else:
        return {"error": "Please provide either image_path or image_url"}
    
    try:
        # Create the chat completion request (older API format)
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Vision model for older API
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Look at this prescription image carefully and extract the following information. Return ONLY a valid JSON object with these exact fields:

                            {
                            "pharmacy_or_doctor_name": "name found in image",
                            "address": "address found in image",
                            "rx_number": "prescription number found",
                            "store_number": "store number found", 
                            "medicines_names":
                            "medicines_names": [
                                {
                                "medicine_name": "individual medication name found",
                                "description": "dosage instructions found",
                                "qty": "quantity found",
                                "side_effects": "side effects mentioned"
                                }
                            ]
                            }


                            IMPORTANT: Extract each medicine separately as individual objects in the medicines_names array.
                            If any field is not visible or found in the image, use "none" as the value.
                            Look for text like "RX Number:", "Store Number:", medication names, dosage instructions, quantities, and any side effect warnings.
                            Return ONLY the JSON object, no additional text or explanation."""
                        },
                        image_content
                    ]
                }
            ],
            max_tokens=800
        )
        
        # Get the response content
        content = response['choices'][0]['message']['content'].strip()
        
        # Clean up the response - remove markdown formatting if present
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        # Try to parse as JSON
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # If response is not valid JSON, create structured response
            print(f"Raw response: {content}")
            return {
                "pharmacy_or_doctor_name": "none",
                "address": "none", 
                "rx_number": "none",
                "store_number": "none",
                "medicines_names": [{
                    "medicine_name": "none",
                    "description": "none",
                    "qty": "none",
                    "side_effects": "none"
                }],
                "raw_response": content
            }
            
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

# Example usage functions
def process_local_image(image_path):
    """Process a local image file"""
    return extract_prescription_info(image_path=image_path)

def process_url_image(image_url):
    """Process an image from URL"""
    return extract_prescription_info(image_url=image_url)

# Example usage
if __name__ == "__main__":
    # Test with the specific image you provided
    test_url = "https://i.pinimg.com/736x/d5/ef/9a/d5ef9a9629dc57e5b75266931235d9a7.jpg"
    
    print("Processing prescription image...")
    result = process_url_image(test_url)
    print("Result:")
    print(json.dumps(result, indent=2))
    
    # Example for local file (uncomment to use)
    # result = process_local_image("prescription_image.jpg")
    # print("Result from local file:")
    # print(json.dumps(result, indent=2))