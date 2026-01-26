"""
Gemini-Powered CV Extractor
Uses Google's Gemini API for intelligent CV extraction with reliable JSON generation
"""

import google.generativeai as genai
import os
import json
from groq import Groq
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GeminiCVExtractor:
    """Extract CV information using Google's Gemini API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Gemini CV Extractor
        
        Args:
            api_key: Google API key (if not provided, will look for GOOGLE_API_KEY env var)
        """
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        #self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        # if not self.api_key:
        #     raise ValueError("Google API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        
        # Configure Gemini
        # genai.configure(api_key=self.api_key)
        
        # Use Gemini 1.5 Flash with relaxed safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        self.groq_model = "openai/gpt-oss-120b"
        # self.model = genai.GenerativeModel(
        #     'gemini-3-flash-preview',
        #     generation_config={
        #         "response_mime_type": "application/json",
        #         "temperature": 0.1
        #     },
        #     safety_settings=safety_settings
        # )
    
    def extract_cv_info(self, raw_text: str) -> Dict[str, Any]:
        """
        Extract structured CV information using Gemini
        
        Args:
            raw_text: Raw text extracted from CV file
            
        Returns:
            Dictionary containing structured CV information
        """
        logger.info("🔷 Gemini extract_cv_info called")
        logger.info(f"📝 Input text length: {len(raw_text)} characters")
        
        prompt = self._create_extraction_prompt(raw_text)
        logger.info(f"📋 Prompt created, length: {len(prompt)} characters")
        
        try:
            logger.info("🚀 Sending request to Groq API...")
            # Generate response with Gemini
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.groq_model,
                temperature=0.1,  # Very low temperature for consistent JSON
                max_tokens=4096,
                response_format={"type": "json_object"}  # Force JSON response
            ).choices[0].message.content
            
            logger.info("✅ Received response from Groq")
            logger.info(f"📊 Response length: {len(response)} characters")
            logger.info(f"📄 First 200 chars of response: {response[:200]}...")
            # response = self.model.generate_content(prompt)
            print(response)
            # Check if response was blocked
            # if not response.candidates:
            #     print("Response blocked by safety filters")
            #     return self._create_fallback_structure(raw_text)
            
            # candidate = response.candidates[0]
            
            # # Check finish reason
            # if candidate.finish_reason != 1:  # 1 = STOP (success)
            #     finish_reason_map = {
            #         2: "MAX_TOKENS",
            #         3: "SAFETY",
            #         4: "RECITATION",
            #         5: "OTHER"
            #     }
            #     reason = finish_reason_map.get(candidate.finish_reason, "UNKNOWN")
            #     print(f"Response generation stopped: {reason}")
                
            #     # If it's not a safety issue, try to get partial content
            #     if candidate.finish_reason != 3 and hasattr(candidate, 'content') and candidate.content.parts:
            #         response_text = candidate.content.parts[0].text
            #     else:
            #         return self._create_fallback_structure(raw_text)
            # else:
            #     # Normal case - get text
            #     if not hasattr(response, 'text') or not response.text:
            #         print("Response has no text content")
            #         return self._create_fallback_structure(raw_text)
                
            #     response_text = response.text
            
            # Clean the response text
            response_text = response.strip()
            logger.info("🧹 Cleaned response text")
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
                logger.info("✂️  Removed ```json prefix")
            if response_text.startswith('```'):
                response_text = response_text[3:]  # Remove ```
                logger.info("✂️  Removed ``` prefix")
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ```
                logger.info("✂️  Removed ``` suffix")
            
            response_text = response_text.strip()
            
            # Try to parse JSON
            logger.info("📊 Attempting to parse JSON...")
            try:
                cv_data = json.loads(response_text)
                logger.info("✅ JSON parsed successfully!")
                logger.info(f"📋 CV data keys: {list(cv_data.keys())}")
            except json.JSONDecodeError as je:
                logger.error(f"❌ JSON decode error: {str(je)}")
                logger.error(f"Response text (first 500 chars): {response_text[:500]}")
                
                logger.info("🔧 Attempting to fix JSON...")
                # Try to fix common JSON issues
                response_text = self._fix_json_string(response_text)
                cv_data = json.loads(response_text)
                logger.info("✅ JSON fixed and parsed!")
            
            # Validate and structure
            logger.info("🔍 Validating and structuring data...")
            result = self._validate_and_structure(cv_data)
            logger.info("✅ Data validated and structured successfully!")
            return result
            
        except Exception as e:
            logger.error("="*60)
            logger.error(f"❌ ERROR IN GEMINI EXTRACTION")
            logger.error("="*60)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Full traceback:")
            import traceback
            logger.error(traceback.format_exc())
            logger.error("="*60)
            logger.info("🔄 Using fallback extraction method...")
            # Return fallback structure
            result = self._create_fallback_structure(raw_text)
            logger.info("✅ Fallback structure created")
            return result
    
    def _create_extraction_prompt(self, raw_text: str) -> str:
        """Create extraction prompt for Gemini"""
        
        # Truncate very long CVs
        if len(raw_text) > 30000:
            raw_text = raw_text[:30000] + "... [truncated]"
        
        # Clean the text to avoid JSON issues
        raw_text = self._clean_text_for_json(raw_text)
        
        prompt = f"""
Extract all information from this CV and return it as JSON following this exact schema:

CV TEXT:
{raw_text}

Return JSON with this structure:
{{
  "personal_info": {{
    "name": "full name or null",
    "email": "email or null",
    "phone": "phone or null",
    "location": "city, state/country or null",
    "linkedin": "LinkedIn URL or null",
    "github": "GitHub URL or null",
    "website": "website URL or null"
  }},
  "professional_summary": "2-3 sentence summary or null",
  "education": [
    {{
      "degree": "degree name",
      "institution": "school name",
      "graduation_year": "year",
      "gpa": "GPA or null",
      "location": "location or null"
    }}
  ],
  "experience": [
    {{
      "title": "job title",
      "company": "company name",
      "location": "location or null",
      "start_date": "start date",
      "end_date": "end date or Present",
      "responsibilities": ["responsibility 1", "responsibility 2"],
      "achievements": ["achievement with metrics"],
      "technologies": ["tech1", "tech2"]
    }}
  ],
  "skills": {{
    "programming_languages": ["Python", "JavaScript"],
    "frameworks": ["React", "Django"],
    "tools": ["Docker", "Git"],
    "databases": ["PostgreSQL"],
    "cloud": ["AWS"],
    "technical": ["other technical skills"],
    "soft_skills": ["Leadership", "Communication"]
  }},
  "projects": [
    {{
      "name": "project name",
      "description": "brief description",
      "technologies": ["tech1", "tech2"],
      "url": "project URL or null"
    }}
  ],
  "certifications": [
    {{
      "name": "certification name",
      "issuer": "issuing organization",
      "date": "date",
      "url": "URL or null"
    }}
  ],
  "awards": [
    {{
      "title": "award title",
      "issuer": "organization",
      "date": "date",
      "description": "description or null"
    }}
  ],
  "languages": [
    {{
      "language": "English",
      "proficiency": "Native/Fluent/Intermediate/Basic"
    }}
  ],
  "total_years_experience": "calculated years like 5 years or null",
  "industry": "primary industry or null"
}}

CRITICAL RULES:
1. Return ONLY valid JSON - no markdown, no code blocks, no extra text
2. Extract ONLY information present in the CV
3. Use null for missing fields
4. Use empty arrays [] for missing lists
5. ESCAPE all quotes in strings - use \\" for quotes inside strings
6. Replace line breaks in strings with spaces
7. Remove any special characters that could break JSON
8. Ensure all strings are properly terminated
9. Do not include any text before or after the JSON object
10. Start with {{ and end with }}
"""
        
        return prompt
    
    def _validate_and_structure(self, cv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and ensure proper structure"""
        
        # Default structure
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
                "databases": [],
                "cloud": [],
                "technical": [],
                "soft_skills": []
            },
            "projects": [],
            "certifications": [],
            "awards": [],
            "languages": [],
            "total_years_experience": None,
            "industry": None
        }
        
        # Merge with default
        for key in default_structure:
            if key not in cv_data:
                cv_data[key] = default_structure[key]
            elif isinstance(default_structure[key], dict):
                for subkey in default_structure[key]:
                    if subkey not in cv_data[key]:
                        cv_data[key][subkey] = default_structure[key][subkey]
        
        return cv_data
    
    def _create_fallback_structure(self, raw_text: str) -> Dict[str, Any]:
        """Create fallback structure if extraction fails - try regex extraction"""
        
        print("Attempting regex-based extraction as fallback...")
        
        # Try to import and use regex extractor
        try:
            import sys
            import os
            
            # Add parsers directory to path if not already there
            parsers_path = os.path.join(os.path.dirname(__file__), 'parsers')
            if parsers_path not in sys.path:
                sys.path.insert(0, parsers_path)
            
            from parsers.cv_extractor import extract_cv_info
            
            # Use regex extraction
            cv_data = extract_cv_info(raw_text)
            cv_data['extraction_method'] = 'regex_fallback'
            return cv_data
            
        except Exception as e:
            print(f"Regex extraction also failed: {str(e)}")
            
            # Return minimal structure
            return {
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
                    "databases": [],
                    "cloud": [],
                    "technical": [],
                    "soft_skills": []
                },
                "projects": [],
                "certifications": [],
                "awards": [],
                "languages": [],
                "total_years_experience": None,
                "industry": None,
                "raw_text_snippet": raw_text[:500],
                "extraction_error": True,
                "extraction_method": "failed"
            }
    
    def _fix_json_string(self, json_str: str) -> str:
        """
        Attempt to fix common JSON issues
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Fixed JSON string
        """
        import re
        
        # Remove any BOM or invisible characters
        json_str = json_str.encode('utf-8', 'ignore').decode('utf-8')
        
        # Remove any trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Try to escape unescaped quotes in string values
        # This is a simple heuristic and may not catch all cases
        lines = json_str.split('\n')
        fixed_lines = []
        
        for line in lines:
            # If line contains a string value with unescaped quotes
            if ':' in line and '"' in line:
                # Split at first colon
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_part = parts[0]
                    value_part = parts[1]
                    
                    # Count quotes in value part
                    if value_part.count('"') % 2 != 0:
                        # Odd number of quotes - try to fix
                        # Look for patterns like "text "word" more text"
                        # and replace with "text \"word\" more text"
                        value_part = re.sub(r'([^\\])"([^",}\]]+)"', r'\1\"\2\"', value_part)
                        line = key_part + ':' + value_part
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _clean_text_for_json(self, text: str) -> str:
        """
        Clean text to prevent JSON parsing issues
        
        Args:
            text: Raw CV text
            
        Returns:
            Cleaned text safe for JSON
        """
        import re
        
        # Replace smart quotes with regular quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove null bytes and other control characters
        text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t'])
        
        # Replace newlines with spaces to avoid multiline string issues
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text.strip()
    
    def extract_with_insights(self, raw_text: str) -> Dict[str, Any]:
        """
        Extract CV info with additional career insights
        
        Args:
            raw_text: Raw CV text
            
        Returns:
            CV data with insights
        """
        # First extract basic info
        cv_data = self.extract_cv_info(raw_text)
        
        # Add insights
        insights_prompt = f"""
Based on this CV data, provide career insights:

{json.dumps(cv_data, indent=2)}

Return JSON with:
{{
  "career_level": "Entry/Mid/Senior/Executive",
  "primary_industry": "main industry",
  "strengths": ["strength1", "strength2", "strength3"],
  "recommended_roles": ["role1", "role2", "role3"],
  "skill_gaps": ["missing skill1", "missing skill2"]
}}
"""
        
        try:
            response = self.model.generate_content(insights_prompt)
            insights = json.loads(response.text)
            cv_data['insights'] = insights
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
        
        return cv_data


# Helper function
def extract_cv_with_gemini(
    raw_text: str,
    api_key: Optional[str] = None,
    include_insights: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to extract CV using Gemini
    
    Args:
        raw_text: Raw CV text
        api_key: Google API key (optional)
        include_insights: Include career insights
        
    Returns:
        Structured CV data
    """
    extractor = GeminiCVExtractor(api_key=api_key)
    
    if include_insights:
        return extractor.extract_with_insights(raw_text)
    else:
        return extractor.extract_cv_info(raw_text)