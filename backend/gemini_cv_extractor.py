"""
Gemini-Powered CV Extractor
Uses Google's Gemini API for intelligent CV extraction with reliable JSON generation
"""

import google.generativeai as genai
import os
import json
from typing import Dict, Any, Optional


class GeminiCVExtractor:
    """Extract CV information using Google's Gemini API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Gemini CV Extractor
        
        Args:
            api_key: Google API key (if not provided, will look for GOOGLE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 1.5 Flash for speed and JSON mode
        self.model = genai.GenerativeModel(
            "gemini-3-flash-preview",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )
    
    def extract_cv_info(self, raw_text: str) -> Dict[str, Any]:
        """
        Extract structured CV information using Gemini
        
        Args:
            raw_text: Raw text extracted from CV file
            
        Returns:
            Dictionary containing structured CV information
        """
        
        prompt = self._create_extraction_prompt(raw_text)
        
        try:
            # Generate response with Gemini
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            cv_data = json.loads(response.text)
            
            # Validate and structure
            return self._validate_and_structure(cv_data)
            
        except Exception as e:
            print(f"Error in Gemini extraction: {str(e)}")
            # Return fallback structure
            return self._create_fallback_structure(raw_text)
    
    def _create_extraction_prompt(self, raw_text: str) -> str:
        """Create extraction prompt for Gemini"""
        
        # Truncate very long CVs
        if len(raw_text) > 30000:
            raw_text = raw_text[:30000] + "... [truncated]"
        
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
  "total_years_experience": "calculated years like '5 years' or null",
  "industry": "primary industry or null"
}}

RULES:
1. Extract ONLY information present in the CV
2. Use null for missing fields
3. Use empty arrays [] for missing lists
4. Identify quantified achievements (numbers, percentages, metrics)
5. Categorize skills properly
6. Calculate total years of experience from work history
7. Return valid JSON only
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
        """Create fallback structure if extraction fails"""
        
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
            "extraction_error": True
        }
    
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