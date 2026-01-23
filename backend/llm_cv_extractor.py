"""
LLM-Powered CV Extractor
Uses Groq's Llama 3.1 to intelligently extract structured information from CVs
"""

from groq import Groq
import os
import json
from typing import Dict, Any, Optional


class LLMCVExtractor:
    """Extract CV information using LLM for better accuracy"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the LLM CV Extractor
        
        Args:
            api_key: Groq API key (if not provided, will look for GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it as parameter.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"
    
    def extract_cv_info(self, raw_text: str) -> Dict[str, Any]:
        """
        Extract structured CV information using LLM
        
        Args:
            raw_text: Raw text extracted from CV file
            
        Returns:
            Dictionary containing structured CV information
        """
        
        prompt = self._create_extraction_prompt(raw_text)
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a CV parser that returns ONLY valid JSON. 
                        Your response must be pure JSON with no additional text, explanations, or markdown formatting.
                        Start with { and end with }. Never include code blocks (```json) or any text outside the JSON object.
                        Extract information accurately from CVs and structure it according to the provided schema."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.1,  # Very low temperature for consistent JSON
                max_tokens=4096,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            response_content = chat_completion.choices[0].message.content
            
            # Parse JSON response
            cv_data = json.loads(response_content)
            
            # Validate and structure the response
            return self._validate_and_structure(cv_data)
            
        except Exception as e:
            print(f"Error in LLM extraction: {str(e)}")
            # Fallback to basic extraction
            return self._create_fallback_structure(raw_text)
    
    def _create_extraction_prompt(self, raw_text: str) -> str:
        """Create a detailed prompt for CV extraction"""
        
        # Truncate very long CVs to avoid token limits
        if len(raw_text) > 15000:
            raw_text = raw_text[:15000] + "... [truncated]"
        
        prompt = f"""
You must extract information from the CV below and return ONLY a valid JSON object. Do not include any explanatory text, markdown formatting, or code blocks - just pure JSON.

CV TEXT:
{raw_text}

Return a JSON object with this EXACT structure (use null for missing data, use empty arrays [] for missing lists):

{{
  "personal_info": {{
    "name": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "location": "string or null",
    "linkedin": "string or null",
    "github": "string or null",
    "website": "string or null"
  }},
  "professional_summary": "string or null",
  "education": [
    {{
      "degree": "string",
      "institution": "string",
      "graduation_year": "string",
      "gpa": "string or null"
    }}
  ],
  "experience": [
    {{
      "title": "string",
      "company": "string",
      "start_date": "string",
      "end_date": "string or Present",
      "responsibilities": ["string"]
    }}
  ],
  "skills": {{
    "programming_languages": ["string"],
    "frameworks": ["string"],
    "tools": ["string"],
    "technical": ["string"],
    "soft_skills": ["string"]
  }},
  "projects": [
    {{
      "name": "string",
      "description": "string",
      "technologies": ["string"]
    }}
  ],
  "certifications": [
    {{
      "name": "string",
      "issuer": "string",
      "date": "string"
    }}
  ],
  "awards": [
    {{
      "title": "string",
      "issuer": "string",
      "date": "string"
    }}
  ]
}}

IMPORTANT EXTRACTION RULES:
1. Return ONLY valid JSON - no text before or after
2. Do not include markdown code blocks or backticks
3. Extract EXACTLY what is in the CV - don't invent information
4. For missing fields, use null or empty array []
5. All field names must match the structure exactly
6. Dates should be strings in format "YYYY" or "Month YYYY"
7. Arrays must contain only the specified types
8. Do not add any commentary or explanations

The response must start with {{ and end with }} - nothing else.
"""
        
        return prompt
    
    def _validate_and_structure(self, cv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and ensure proper structure of extracted data"""
        
        # Define default structure (simplified)
        default_structure = {
            "personal_info": {
                "name": None,
                "email": None,
                "phone": None,
                "location": None,
                "linkedin": None,
                "github": None,
                "website": None
            },
            "professional_summary": None,
            "education": [],
            "experience": [],
            "skills": {
                "programming_languages": [],
                "frameworks": [],
                "tools": [],
                "technical": [],
                "soft_skills": []
            },
            "projects": [],
            "certifications": [],
            "awards": []
        }
        
        # Merge with default structure
        for key in default_structure:
            if key not in cv_data:
                cv_data[key] = default_structure[key]
            elif isinstance(default_structure[key], dict):
                # Merge nested dictionaries
                for subkey in default_structure[key]:
                    if subkey not in cv_data[key]:
                        cv_data[key][subkey] = default_structure[key][subkey]
        
        return cv_data
    
    def _create_fallback_structure(self, raw_text: str) -> Dict[str, Any]:
        """Create a basic structure if LLM extraction fails"""
        
        return {
            "personal_info": {
                "name": None,
                "email": None,
                "phone": None,
                "location": None,
                "linkedin": None,
                "github": None,
                "website": None,
                "portfolio": None
            },
            "professional_summary": None,
            "education": [],
            "experience": [],
            "skills": {
                "technical": [],
                "programming_languages": [],
                "frameworks": [],
                "tools": [],
                "databases": [],
                "cloud": [],
                "soft_skills": [],
                "languages": []
            },
            "projects": [],
            "certifications": [],
            "publications": [],
            "awards": [],
            "volunteer": [],
            "interests": [],
            "total_years_experience": None,
            "industry": None,
            "raw_text": raw_text[:500],  # Include snippet for debugging
            "extraction_error": True
        }
    
    def extract_specific_section(
        self,
        raw_text: str,
        section: str
    ) -> Dict[str, Any]:
        """
        Extract a specific section from CV
        
        Args:
            raw_text: Raw CV text
            section: Section to extract (e.g., 'experience', 'education', 'skills')
            
        Returns:
            Extracted section data
        """
        
        prompt = f"""
Extract ONLY the {section} section from this CV text and return it in JSON format.

CV TEXT:
{raw_text}

Return a JSON object with the {section} data following the standard CV format.
If the section is not found, return an empty structure.
"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a CV parser. Extract only the requested section accurately."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            return json.loads(chat_completion.choices[0].message.content)
            
        except Exception as e:
            print(f"Error extracting {section}: {str(e)}")
            return {}
    
    def enhance_extraction(
        self,
        basic_data: Dict[str, Any],
        raw_text: str
    ) -> Dict[str, Any]:
        """
        Enhance basic extraction with additional analysis
        
        Args:
            basic_data: Basic extracted CV data
            raw_text: Original CV text
            
        Returns:
            Enhanced CV data with additional insights
        """
        
        prompt = f"""
Analyze this CV and provide additional insights:

CV DATA:
{json.dumps(basic_data, indent=2)}

ORIGINAL TEXT:
{raw_text}

Return JSON with:
{{
  "strengths": ["strength1", "strength2"],
  "gaps": ["gap1", "gap2"],
  "career_level": "Entry/Mid/Senior/Executive",
  "primary_industry": "Industry name",
  "key_achievements": ["achievement1", "achievement2"],
  "recommended_job_titles": ["title1", "title2"],
  "skills_to_highlight": ["skill1", "skill2"]
}}
"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a career analyst providing CV insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.5,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            insights = json.loads(chat_completion.choices[0].message.content)
            basic_data['insights'] = insights
            
            return basic_data
            
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            return basic_data


# Helper function for easy use
def extract_cv_with_llm(
    raw_text: str,
    api_key: Optional[str] = None,
    include_insights: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to extract CV information using LLM
    
    Args:
        raw_text: Raw text from CV
        api_key: Groq API key (optional)
        include_insights: Whether to include additional insights
        
    Returns:
        Structured CV data
    """
    extractor = LLMCVExtractor(api_key=api_key)
    cv_data = extractor.extract_cv_info(raw_text)
    
    if include_insights:
        cv_data = extractor.enhance_extraction(cv_data, raw_text)
    
    return cv_data