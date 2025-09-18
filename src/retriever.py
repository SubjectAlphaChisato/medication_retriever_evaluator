"""
Retriever agent for extracting structured hepatotoxicity information from LiverTox XML documents.
"""

import json
import re
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from openai import OpenAI
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from .models import HepatotoxicityData, ExtractionResult, RiskFactor, Dose, OnsetTime
from .xml_parser import LiverToxXMLParser
from .config import DEFAULT_RETRIEVER_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY, MAX_RETRIES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HepatotoxicityRetriever:
    """Agent for extracting structured hepatotoxicity data from XML documents."""
    
    def __init__(self, model: str = DEFAULT_RETRIEVER_MODEL):
        """Initialize the retriever with LLM client."""
        self.model = model or DEFAULT_RETRIEVER_MODEL
        logger.info(f"Initializing retriever with model: {self.model}")

        # Initialize appropriate client based on model
        if self.model and self.model.startswith("claude"):
            if not ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found")
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
            self.provider = "anthropic"
        else:
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.provider = "openai"
            # Default to GPT-4o if model is None or not recognized
            if not self.model or not self.model.startswith("gpt"):
                self.model = "gpt-4o"
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_llm(self, messages: list) -> str:
        """Call LLM API with retry logic."""
        try:
            if self.provider == "anthropic":
                # Convert OpenAI-style messages to Anthropic format
                system_message = ""
                user_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    elif msg["role"] == "user":
                        user_messages.append(msg["content"])
                
                combined_user_message = "\n\n".join(user_messages)
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.1,
                    system=system_message,
                    messages=[{"role": "user", "content": combined_user_message}]
                )
                return response.content[0].text
            else:
                # OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling {self.provider} API ({self.model}): {e}")
            raise
    
    def extract_hepatotoxicity_data(self, xml_file_path: str) -> ExtractionResult:
        """Extract hepatotoxicity data from a LiverTox XML file."""
        # Parse XML
        parser = LiverToxXMLParser(xml_file_path)
        drug_name = parser.get_drug_name()
        
        # Get structured content
        sections = parser.extract_sections()
        full_text = parser.extract_text_content()
        
        # Create extraction prompt
        prompt = self._create_extraction_prompt(drug_name, sections, full_text)
        
        # Call LLM
        messages = [
            {
                "role": "system",
                "content": """You are an expert medical information extraction specialist. Your task is to extract structured hepatotoxicity information from LiverTox XML documents. 

You must:
1. Extract only information that is explicitly present in the document
2. Be conservative - if information is unclear or not present, mark as null/unknown
3. Always include source information for traceability
4. Return valid JSON that matches the expected schema
"""
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        response_text = self._call_llm(messages)
        
        # Parse response
        try:
            # Extract JSON from response if wrapped in markdown
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = response_text
            
            extracted_data = json.loads(json_text)
            
            # Clean up "null" strings to actual None values
            for key, value in extracted_data.items():
                if value == "null" or value == "None":
                    extracted_data[key] = None
                # Handle boolean field specifically
                elif key == "is_immune_mediated" and value is None:
                    extracted_data[key] = False  # Default to False for boolean field
            
            # Convert to Pydantic model
            hepatotoxicity_data = HepatotoxicityData(**extracted_data)
            
            # Create result
            result = ExtractionResult(
                data=hepatotoxicity_data,
                extraction_timestamp=datetime.now().isoformat(),
                model_used=self.model,
                extraction_notes=f"Extracted from {Path(xml_file_path).name}"
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"Error creating HepatotoxicityData: {e}")
            if 'extracted_data' in locals():
                logger.error(f"Extracted data: {extracted_data}")
            raise
    
    def _create_extraction_prompt(self, drug_name: str, sections: Dict[str, str], full_text: str) -> str:
        """Create the extraction prompt for the LLM."""
        
        # Limit text size to avoid token limits
        max_text_length = 15000
        if len(full_text) > max_text_length:
            full_text = full_text[:max_text_length] + "... [truncated]"
        
        # Create sections summary
        sections_text = ""
        for section_name, content in sections.items():
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"
            sections_text += f"\n## {section_name}\n{content}\n"
        
        return f"""
Extract hepatotoxicity information for the drug "{drug_name}" from the following LiverTox document.

**DOCUMENT SECTIONS:**
{sections_text}

**FULL DOCUMENT TEXT (for reference):**
{full_text}

**EXTRACTION REQUIREMENTS:**

Extract the following information and return as JSON:

1. **drug_name** (required): "{drug_name}"

2. **dili_likelihood_score** (required): Based on the evidence in the document, assign one of [A, B, C, D, E, E*, X]:
   - Look for explicit likelihood scores or category statements
   - If not explicit, assess based on number of cases, evidence quality, and certainty of causation
   - Consider phrases like "well established", "known cause", "rare reports", "no evidence"

3. **injury_pattern**: Look for terms like hepatocellular, cholestatic, mixed, intrinsic, idiosyncratic
   
4. **fraction_patients_with_enzyme_elevation**: Look for percentages or fractions of patients with elevated enzymes

5. **fraction_patients_with_dili**: Look for percentages or fractions with clinical liver injury

6. **risk_factors**: Extract any mentioned risk factors with supporting quotes

7. **safe_dose**: Look for recommended therapeutic doses

8. **toxic_dose**: Look for doses associated with toxicity or overdose

9. **onset_time**: Time from exposure to injury onset

10. **peak_alt**, **peak_alp**, **r_ratio**, **bilirubin_peak**: Look for specific lab values

11. **is_immune_mediated**: Look for mentions of immune response, hypersensitivity, autoantibodies

12. **regulatory_status**: Look for approval status, withdrawal, investigational use

13. **extraction_sources**: For each field you extract, note the section/source where you found it

**OUTPUT FORMAT:**
Return ONLY valid JSON matching this schema:

```json
{{
    "drug_name": "string",
    "dili_likelihood_score": "A|B|C|D|E|E*|X",
    "injury_pattern": "hepatocellular|cholestatic|mixed|intrinsic|idiosyncratic|unclear",
    "fraction_patients_with_enzyme_elevation": 0.0,
    "fraction_patients_with_dili": 0.0,
    "risk_factors": [
        {{"factor": "string", "supporting_quote": "string"}}
    ],
    "safe_dose": {{"value": 0.0, "unit": "string", "frequency": "string"}},
    "toxic_dose": {{"value": 0.0, "unit": "string", "frequency": "string"}},
    "onset_time": {{"min": 0, "max": 0, "typical": 0, "unit": "string"}},
    "peak_alt": 0.0,
    "peak_alp": 0.0,
    "r_ratio": 0.0,
    "bilirubin_peak": 0.0,
    "is_immune_mediated": false,
    "regulatory_status": "approved|investigational|withdrawn|unregulated",
    "extraction_sources": {{
        "field_name": "source_section_or_context"
    }}
}}
```

**CRITICAL RULES:**
- Use JSON null for missing information, NOT the string "null"
- If no information is available for a field, use null (not "null")
- For required fields like drug_name, always provide a value
- For boolean fields, use true/false (not "true"/"false")

**IMPORTANT:**
- Use null for any field where information is not available
- Be conservative - only extract what is clearly stated
- Include specific quotes and sources for traceability
"""
