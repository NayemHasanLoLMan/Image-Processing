import os
import json
import base64
import openai
from openai import OpenAI
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Medicine:
    """Data class for medicine information"""
    medicine_name: str = "none"
    description: str = "none"
    qty: str = "none"
    side_effects: str = "none"

@dataclass
class PrescriptionData:
    """Data class for prescription information"""
    pharmacy_or_doctor_name: str = "none"
    title_or_doctor_details: str = "none"
    contact_details: str = "none"
    date: str = "none"
    address: str = "none"
    rx_number: str = "none"
    store_number: str = "none"
    medicines_names: List[Dict] = None
    
    def __post_init__(self):
        if self.medicines_names is None:
            self.medicines_names = []

class PrescriptionExtractor:
    """Main class for extracting prescription information from images"""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.5):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.existing_data = PrescriptionData()
        
    def _get_system_prompt(self) -> Dict[str, str]:
        """Get the system prompt for the AI assistant"""
        return {
            "role": "system",
            "content": (
                "You are a medical assistant that extracts structured data from prescription images. "
                "Respond only with valid JSON based on the user's instructions. "
                "Be thorough and accurate in your extraction."
            )
        }
    
    def _get_user_instruction(self, context_data: Optional[PrescriptionData] = None) -> str:
        """Generate dynamic user instruction based on existing context"""
        base_instruction = """
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
            "description": "dosage instructions. If vague or missing, infer from context. Reconstruct clearly using medical knowledge. For example: 'Take one (1x) tablet in the morning and one (1x) at night for five (5) days.'",
            "qty": "quantity found. If not present, estimate based on dosage duration. For example: '60 tablets for 2 per day for 30 days'.",
            "side_effects": "any mentioned side effects. If missing, include general known side effects for the medicine."
            }
        ]
        }
        """
        
        if context_data and self._has_existing_data(context_data):
            context_instruction = f"""
            
            CONTEXT: You have access to previously extracted information:
            {json.dumps(asdict(context_data), indent=2)}
            
            INSTRUCTIONS:
            - If the new image contains information that complements or updates the existing data, merge them intelligently
            - If you find conflicting information, prioritize the information from the current image
            - If a field is missing in the current image but exists in context, preserve the context value
            - For medicines, add new medicines to the existing list, but avoid duplicates
            - Update existing medicine information if the current image provides more detail
            """
            return base_instruction + context_instruction
        
        return base_instruction + """
        
        STRICT RULES:
        - Each medicine must be a separate object in the medicines_names array
        - Use the word "none" for any field not visible or inferable from the image
        - Use clear and full-sentence structure for `description`, `qty`, and `side_effects`
        - Do NOT output anything except the JSON object. No commentary, no explanations
        """
    
    def _has_existing_data(self, data: PrescriptionData) -> bool:
        """Check if there's meaningful existing data"""
        fields_to_check = [
            data.pharmacy_or_doctor_name, data.title_or_doctor_details,
            data.contact_details, data.date, data.address,
            data.rx_number, data.store_number
        ]
        return any(field != "none" and field for field in fields_to_check) or len(data.medicines_names) > 0
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise
    
    def _create_image_content(self, image_path: Optional[str] = None, 
                            image_url: Optional[str] = None) -> Optional[Dict]:
        """Create image content for API request"""
        if image_path:
            base64_image = self._encode_image(image_path)
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        elif image_url:
            return {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        return None
    
    def _parse_response(self, content: str) -> Dict:
        """Parse and clean the API response"""
        # Remove markdown formatting
        content = content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {content}")
            return self._get_default_response()
    
    def _get_default_response(self) -> Dict:
        """Get default response structure when parsing fails"""
        return asdict(PrescriptionData(medicines_names=[asdict(Medicine())]))
    
    def _merge_prescription_data(self, new_data: Dict, 
                               existing_data: PrescriptionData) -> PrescriptionData:
        """Intelligently merge new data with existing data"""
        merged = PrescriptionData()
        
        # Merge basic fields (prioritize new data if not "none")
        for field in ['pharmacy_or_doctor_name', 'title_or_doctor_details', 
                     'contact_details', 'date', 'address', 'rx_number', 'store_number']:
            new_value = new_data.get(field, "none")
            existing_value = getattr(existing_data, field, "none")
            
            if new_value != "none":
                setattr(merged, field, new_value)
            elif existing_value != "none":
                setattr(merged, field, existing_value)
            else:
                setattr(merged, field, "none")
        
        # Merge medicines (more complex logic)
        merged.medicines_names = self._merge_medicines(
            new_data.get('medicines_names', []),
            existing_data.medicines_names
        )
        
        return merged
    
    def _merge_medicines(self, new_medicines: List[Dict], 
                        existing_medicines: List[Dict]) -> List[Dict]:
        """Merge medicine lists, avoiding duplicates and updating existing entries"""
        merged_medicines = []
        existing_names = {med.get('medicine_name', '').lower().strip() 
                         for med in existing_medicines if med.get('medicine_name') != "none"}
        
        # Add existing medicines
        for med in existing_medicines:
            if med.get('medicine_name') != "none":
                merged_medicines.append(med)
        
        # Add or update with new medicines
        for new_med in new_medicines:
            med_name = new_med.get('medicine_name', '').lower().strip()
            if med_name and med_name != "none":
                # Check if medicine already exists
                existing_index = None
                for i, existing_med in enumerate(merged_medicines):
                    if existing_med.get('medicine_name', '').lower().strip() == med_name:
                        existing_index = i
                        break
                
                if existing_index is not None:
                    # Update existing medicine with new information
                    for key, value in new_med.items():
                        if value != "none" and value:
                            merged_medicines[existing_index][key] = value
                else:
                    # Add new medicine
                    merged_medicines.append(new_med)
        
        return merged_medicines if merged_medicines else [asdict(Medicine())]
    
    def extract_from_image(self, image_path: Optional[str] = None, 
                          image_url: Optional[str] = None,
                          update_existing: bool = True) -> Dict:
        """Extract prescription information from image"""
        image_content = self._create_image_content(image_path, image_url)
        if not image_content:
            return {"error": "Please provide either image_path or image_url"}
        
        try:
            context_data = self.existing_data if update_existing else None
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    self._get_system_prompt(),
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._get_user_instruction(context_data)},
                            image_content
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            parsed_data = self._parse_response(content)
            
            if update_existing:
                self.existing_data = self._merge_prescription_data(parsed_data, self.existing_data)
                return asdict(self.existing_data)
            else:
                return parsed_data
                
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return {"error": f"API call failed: {str(e)}"}
    
    def get_current_data(self) -> Dict:
        """Get current accumulated prescription data"""
        return asdict(self.existing_data)
    
    def reset_data(self):
        """Reset accumulated data"""
        self.existing_data = PrescriptionData()
    
    def save_data(self, filename: str):
        """Save current data to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(asdict(self.existing_data), f, indent=2)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
    
    def load_data(self, filename: str):
        """Load data from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.existing_data = PrescriptionData(**data)
            logger.info(f"Data loaded from {filename}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")

# Utility functions for backward compatibility and ease of use
def create_extractor(model: str = "gpt-4o", temperature: float = 0.5) -> PrescriptionExtractor:
    """Create a new prescription extractor instance"""
    return PrescriptionExtractor(model=model, temperature=temperature)

def extract_from_local_image(image_path: str, extractor: Optional[PrescriptionExtractor] = None) -> Dict:
    """Extract prescription info from local image"""
    if extractor is None:
        extractor = create_extractor()
    return extractor.extract_from_image(image_path=image_path, update_existing=False)

def extract_from_url_image(image_url: str, extractor: Optional[PrescriptionExtractor] = None) -> Dict:
    """Extract prescription info from URL image"""
    if extractor is None:
        extractor = create_extractor()
    return extractor.extract_from_image(image_url=image_url, update_existing=False)

# Example usage
if __name__ == "__main__":
    # Create extractor instance
    extractor = create_extractor()
    
    # Process multiple images to build complete prescription data
    test_url = "https://i.pinimg.com/736x/d5/ef/9a/d5ef9a9629dc57e5b75266931235d9a7.jpg"
    
    print("Processing first prescription image...")
    result1 = extractor.extract_from_image(image_url=test_url)
    print("Result 1:")
    print(json.dumps(result1, indent=2))
    
    # You can process additional images to add/update information
    # result2 = extractor.extract_from_image(image_path="second_prescription.jpg")
    
    # Get accumulated data
    print("\nCurrent accumulated data:")
    print(json.dumps(extractor.get_current_data(), indent=2))
    
    # Save data
    extractor.save_data("prescription_data.json")