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

# System prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a medical assistant that extracts structured data from prescription images. "
        "Respond only with valid JSON based on the user's instructions."
    )
}

# Instruction for the model
USER_INSTRUCTION_TEXT = """
You are shown a prescription image. Extract detailed and structured medical information and return ONLY a well-formatted JSON object using the exact schema below:

{
  "pharmacy_or_doctor_name": "name of the doctor or pharmacy as seen in the image",
  "contact_details": "phone number, email, or any other contact information",
  "date_filled": "date of the prescription filled if visible",
  "date_expired": "date of the prescription expiration if visible",
  "address": "address found in the image",
  "rx_number": "prescription number found",
  "store_number": "store number found",
  "medicines_names": [
    {
      "medicine_name": "individual medicine name found",
      "generic_name": "only generic name if available, otherwise 'none'.",
      "instructions": "dosage instructions. If vague or missing, infer from context. Reconstruct clearly using medical knowledge. For example: 'Take one (1x) tablet in the morning and one  (1x) at night for five (5) days.'",
      "qty": "quantity found. If not present, estimate based on dosage duration. (only number)",
      "refills_info": "refill information if available, otherwise 'none'",
      "side_effects": "any mentioned side effects."

    }
  ]
}

STRICT RULES:
- Each medicine must be a separate object in the medicines_names array.
- Use the word "none" for any field not visible or inferable from the image.
- Use clear and full-sentence structure for `description`, `qty`, and `side_effects`. Do not output vague fragments.
- Do NOT output anything except the JSON object. No commentary, no explanations.
"""

def get_image_content(image_paths=None, image_urls=None):
    """Returns a list of OpenAI-compatible image content blocks"""
    image_contents = []

    if image_paths:
        for path in image_paths:
            with open(path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
    elif image_urls:
        for url in image_urls:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            })

    return image_contents if image_contents else None

def extract_prescription_info(image_paths=None, image_urls=None):
    """Extracts structured prescription info from multiple images"""
    image_contents = get_image_content(image_paths=image_paths, image_urls=image_urls)
    if not image_contents:
        return {"error": "Please provide image_paths or image_urls"}

    try:
        user_content = [{"type": "text", "text": USER_INSTRUCTION_TEXT}] + image_contents

        response = openai.ChatCompletion.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                SYSTEM_PROMPT,
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            max_tokens=2000
        )

        content = response['choices'][0]['message']['content'].strip()

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
                "contact_details": "none",
                "date_filled": "none",
                "date_expired" : "none",
                "address": "none",
                "rx_number": "none",
                "store_number": "none",
                "medicines_names": [
                    {
                        "medicine_name": "none",
                        "generic_name": "none",
                        "instructions": "none",
                        "qty": "none",
                        "refills_info": "none",
                        "side_effects": "none"
                    }
                ],
            }

    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

# Utility function
def process_local_images(image_paths):
    return extract_prescription_info(image_paths=image_paths)

# Example usage
if __name__ == "__main__":
    test_paths = [
        'D:/kevon-imageprocessing/iamge/1.jpg',
        'D:/kevon-imageprocessing/iamge/2.jpg',
        'D:/kevon-imageprocessing/iamge/3.jpg'
    ]

    print("Processing all prescription images together...")
    result = process_local_images(test_paths)
    print("Combined Result:")
    print(json.dumps(result, indent=2))
