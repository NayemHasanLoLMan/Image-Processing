import os
import json
import base64
import openai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
MODEL = "gpt-4o"
TEMPERATURE = 0.5  # Low temperature for deterministic output

# System prompt to instruct the assistant
SYSTEM_PROMPT = {
                "role": "system",
                "content": "You are a medical assistant that extracts structured data from prescription images. "
                        "Respond only with valid JSON based on the user's instructions."
            }

USER_INSTRUCTION_TEXT = """
                You are shown a prescription image. Extract detailed and structured medical information and return ONLY a well-formatted JSON object using the exact schema below:

                {
                "pharmacy_or_doctor_name": "name of the doctor or pharmacy as seen in the image",
                "title_or_doctor_details": "title or medical qualifications of the doctor",
                "contact_details": "phone number, email, or any other contact information",
                "date": "date of the prescription if visible",
                "address": "address found in the image",
                "rx_number": "prescription number found",
                "store_number": "store number found",
                "medicines_names": [
                    {
                    "medicine_name": "individual medicine name found",
                    "description": "dosage instructions. If vague or missing, infer from context. Reconstruct clearly using medical knowledge. For example: 'Take one (1x) tablet in the morning and one  (1x) at night for five (5) days.'",
                    "qty": "quantity found. If not present, estimate based on dosage duration. For example: '60 tablets' for '2 per day for 30 days'.",
                    "side_effects": "any mentioned side effects. If missing, include general known side effects for the medicine. For example: 'May cause drowsiness, dizziness, and dry mouth.Contact your healthcare provider right away if you experience any serious side effects'"
                    }
                ]
                }

                STRICT RULES:
                - Each medicine must be a separate object in the medicines_names array.
                - Use the word "none" for any field not visible or inferable from the image.
                - Use clear and full-sentence structure for `description`, `qty`, and `side_effects`. Do not output vague fragments.
                - Do NOT output anything except the JSON object. No commentary, no explanations.
                """

def get_image_content(image_path=None, image_url=None):
    """Returns OpenAI-compatible image content block"""
    if image_path:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    elif image_url:
        return {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }
    else:
        return None

def extract_prescription_info(image_path=None, image_url=None):
    """Extracts structured prescription info from an image"""
    image_content = get_image_content(image_path, image_url)
    if not image_content:
        return {"error": "Please provide either image_path or image_url"}

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                SYSTEM_PROMPT,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_INSTRUCTION_TEXT},
                        image_content
                    ]
                }
            ],
            max_tokens=1500
        )

        content = response['choices'][0]['message']['content'].strip()

        # Remove any markdown
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"Raw response (not JSON): {content}")
            return {
                "pharmacy_or_doctor_name": "none",
                "title_or_doctor_details": "none",
                "contact_details": "none",
                "date": "none",
                "address": "none",
                "rx_number": "none",
                "store_number": "none",
                "medicines_names": [
                    {
                        "medicine_name": "none",
                        "description": "none",
                        "qty": "none",
                        "side_effects": "none"
                    }
                ],
            }

    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

# Utility functions
def process_local_image(image_path):
    return extract_prescription_info(image_path=image_path)

def process_url_image(image_url):
    return extract_prescription_info(image_url=image_url)

# Example usage
if __name__ == "__main__":
    test_url = "https://miro.medium.com/v2/resize:fit:480/1*bV170ZH7Sn_doc5bbyRYWg.jpeg"
    print("Processing prescription image...")
    result = process_url_image(test_url)
    print("Result:")
    print(json.dumps(result, indent=2))
