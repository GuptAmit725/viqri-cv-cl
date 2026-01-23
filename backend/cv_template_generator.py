"""
AI CV Template Generator
Uses Groq's Llama 3.1 to generate optimized CV templates based on job and location
"""

from groq import Groq
import os
import json
from typing import Dict, Any, Optional


class CVTemplateGenerator:
    """Generate AI-powered CV templates optimized for specific jobs and locations"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the CV Template Generator
        
        Args:
            api_key: Groq API key (if not provided, will look for GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it as parameter.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"  # Using Llama 3.1 70B for best results
    
    def generate_template(
        self,
        cv_data: Dict[str, Any],
        target_job: str,
        target_location: str,
        industry: Optional[str] = None,
        experience_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an optimized CV template based on job and location
        
        Args:
            cv_data: Parsed CV data from the extractor
            target_job: Target job title/role
            target_location: Target location
            industry: Target industry (optional)
            experience_level: Experience level (entry, mid, senior) (optional)
            
        Returns:
            Dictionary containing optimized CV template and recommendations
        """
        
        # Create prompt for Llama
        prompt = self._create_prompt(cv_data, target_job, target_location, industry, experience_level)
        
        # Call Groq API
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert CV/resume writer and career consultant with 20+ years of experience. 
                        You specialize in creating ATS-optimized, industry-specific CVs that get interviews. 
                        You understand regional preferences, cultural differences, and industry standards across different locations.
                        Provide detailed, actionable advice in JSON format."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=4096,
                top_p=1,
                stream=False
            )
            
            # Parse response
            response_content = chat_completion.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                template_data = json.loads(response_content)
            except json.JSONDecodeError:
                # If response is not JSON, structure it
                template_data = {
                    "recommendations": response_content,
                    "success": True
                }
            
            return template_data
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_prompt(
        self,
        cv_data: Dict[str, Any],
        target_job: str,
        target_location: str,
        industry: Optional[str],
        experience_level: Optional[str]
    ) -> str:
        """Create a detailed prompt for the LLM"""
        
        # Extract key information from CV data
        personal_info = cv_data.get('personal_info', {})
        skills = cv_data.get('skills', {})
        experience = cv_data.get('experience', [])
        education = cv_data.get('education', [])
        
        prompt = f"""
Based on the following CV information, create an optimized CV template and provide recommendations for a job application.

TARGET JOB: {target_job}
TARGET LOCATION: {target_location}
{f"TARGET INDUSTRY: {industry}" if industry else ""}
{f"EXPERIENCE LEVEL: {experience_level}" if experience_level else ""}

CURRENT CV DATA:
- Name: {personal_info.get('name', 'Not provided')}
- Email: {personal_info.get('email', 'Not provided')}
- Location: {personal_info.get('location', 'Not provided')}
- Skills: {', '.join(skills.get('languages', []) + skills.get('tools', [])[:10])}
- Years of Experience: {len(experience)} roles
- Education: {len(education)} degrees/certifications

EXPERIENCE SUMMARY:
{json.dumps(experience[:3], indent=2) if experience else "No experience listed"}

TASK:
Generate a comprehensive CV optimization strategy. Return your response as a JSON object with the following structure:

{{
  "template_structure": {{
    "format": "recommended format (chronological/functional/hybrid)",
    "sections": ["ordered list of sections to include"],
    "section_priorities": {{"section_name": "why it should be included"}},
    "length": "recommended page length"
  }},
  "content_recommendations": {{
    "summary": "optimized professional summary for this role (2-3 sentences)",
    "key_skills": ["top 8-10 skills to highlight for this role"],
    "experience_tips": ["how to reframe experience for this role"],
    "missing_skills": ["skills commonly required for this role that are missing"],
    "keywords": ["ATS keywords to include for this role and location"]
  }},
  "location_specific": {{
    "format_preferences": "CV format preferences in {target_location}",
    "cultural_considerations": ["important cultural/regional considerations"],
    "common_requirements": ["what employers in {target_location} typically expect"]
  }},
  "industry_insights": {{
    "trends": "current trends in {industry or 'this industry'}",
    "sought_after_skills": ["top skills employers are looking for"],
    "red_flags": ["things to avoid in CV for this industry"]
  }},
  "action_items": {{
    "immediate": ["3-5 changes to make right now"],
    "important": ["5-7 important improvements"],
    "nice_to_have": ["3-5 optional enhancements"]
  }},
  "template_example": {{
    "professional_summary": "example summary tailored to the role",
    "key_achievements": ["3 example achievement bullets formatted properly"],
    "skills_presentation": "how to present skills section"
  }}
}}

Ensure all recommendations are specific, actionable, and optimized for both ATS systems and human recruiters.
"""
        
        return prompt
    
    def generate_job_match_score(
        self,
        cv_data: Dict[str, Any],
        job_description: str
    ) -> Dict[str, Any]:
        """
        Analyze how well the CV matches a specific job description
        
        Args:
            cv_data: Parsed CV data
            job_description: Target job description
            
        Returns:
            Match score and detailed analysis
        """
        
        prompt = f"""
Analyze how well this CV matches the following job description.

JOB DESCRIPTION:
{job_description}

CV DATA:
{json.dumps(cv_data, indent=2)}

Provide a detailed match analysis in JSON format:
{{
  "match_score": <number 0-100>,
  "strengths": ["what matches well"],
  "gaps": ["what's missing"],
  "recommendations": ["how to improve the match"],
  "keyword_analysis": {{
    "matched": ["keywords present in CV"],
    "missing": ["important keywords missing from CV"]
  }}
}}
"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert ATS analyzer and recruitment specialist."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.5,
                max_tokens=2048
            )
            
            response = chat_completion.choices[0].message.content
            return json.loads(response)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def suggest_improvements(
        self,
        cv_data: Dict[str, Any],
        focus_area: str = "general"
    ) -> Dict[str, Any]:
        """
        Get general improvement suggestions for the CV
        
        Args:
            cv_data: Parsed CV data
            focus_area: Area to focus on (general, technical, leadership, etc.)
            
        Returns:
            Improvement suggestions
        """
        
        prompt = f"""
Review this CV and provide improvement suggestions focusing on: {focus_area}

CV DATA:
{json.dumps(cv_data, indent=2)}

Provide actionable suggestions in JSON format:
{{
  "overall_assessment": "brief overall assessment",
  "strong_points": ["what's working well"],
  "improvement_areas": ["what needs work"],
  "specific_suggestions": [
    {{
      "area": "section name",
      "issue": "what's wrong",
      "suggestion": "how to fix it",
      "example": "example of improved version"
    }}
  ],
  "quick_wins": ["easy changes that will have big impact"]
}}
"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional CV writer and career coach."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=2048
            )
            
            response = chat_completion.choices[0].message.content
            return json.loads(response)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Helper function for easy use
def generate_cv_template(
    cv_data: Dict[str, Any],
    target_job: str,
    target_location: str,
    industry: Optional[str] = None,
    experience_level: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate CV template
    
    Args:
        cv_data: Parsed CV data
        target_job: Target job title
        target_location: Target location
        industry: Target industry (optional)
        experience_level: Experience level (optional)
        api_key: Groq API key (optional)
        
    Returns:
        Generated template and recommendations
    """
    generator = CVTemplateGenerator(api_key=api_key)
    return generator.generate_template(
        cv_data=cv_data,
        target_job=target_job,
        target_location=target_location,
        industry=industry,
        experience_level=experience_level
    )