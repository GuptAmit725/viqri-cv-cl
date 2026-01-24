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
        self.model = "llama-3.1-8b-instant"  # Using Llama 3.1 8B for best results
    
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
        
        # Extract ALL information from CV data
        personal_info = cv_data.get('personal_info', {})
        skills = cv_data.get('skills', {})
        experience = cv_data.get('experience', [])
        education = cv_data.get('education', [])
        projects = cv_data.get('projects', [])
        certifications = cv_data.get('certifications', [])
        awards = cv_data.get('awards', [])
        summary = cv_data.get('professional_summary', '')
        total_exp = cv_data.get('total_years_experience', 'Not specified')
        
        # Compile all skills
        all_skills = []
        if skills:
            all_skills.extend(skills.get('programming_languages', []))
            all_skills.extend(skills.get('frameworks', []))
            all_skills.extend(skills.get('tools', []))
            all_skills.extend(skills.get('technical', []))
            all_skills.extend(skills.get('databases', []))
            all_skills.extend(skills.get('cloud', []))
        
        prompt = f"""
Based on the COMPLETE CV information below, create an optimized CV template and provide comprehensive recommendations.

TARGET JOB: {target_job}
TARGET LOCATION: {target_location}
{f"TARGET INDUSTRY: {industry}" if industry else ""}
{f"EXPERIENCE LEVEL: {experience_level}" if experience_level else ""}

COMPLETE CURRENT CV DATA:

PERSONAL INFORMATION:
- Name: {personal_info.get('name', 'Not provided')}
- Email: {personal_info.get('email', 'Not provided')}
- Phone: {personal_info.get('phone', 'Not provided')}
- Location: {personal_info.get('location', 'Not provided')}
- LinkedIn: {personal_info.get('linkedin', 'Not provided')}
- GitHub: {personal_info.get('github', 'Not provided')}
- Total Experience: {total_exp}

PROFESSIONAL SUMMARY:
{summary if summary else "Not provided"}

ALL WORK EXPERIENCE ({len(experience)} roles):
{json.dumps(experience, indent=2) if experience else "No experience listed"}

EDUCATION ({len(education)} degrees):
{json.dumps(education, indent=2) if education else "No education listed"}

COMPLETE SKILLS:
- Programming Languages: {', '.join(skills.get('programming_languages', []))}
- Frameworks: {', '.join(skills.get('frameworks', []))}
- Tools: {', '.join(skills.get('tools', []))}
- Databases: {', '.join(skills.get('databases', []))}
- Cloud: {', '.join(skills.get('cloud', []))}
- Technical: {', '.join(skills.get('technical', []))}
- Soft Skills: {', '.join(skills.get('soft_skills', []))}

PROJECTS ({len(projects)}):
{json.dumps(projects, indent=2) if projects else "No projects listed"}

CERTIFICATIONS ({len(certifications)}):
{json.dumps(certifications, indent=2) if certifications else "No certifications listed"}

AWARDS & ACHIEVEMENTS ({len(awards)}):
{json.dumps(awards, indent=2) if awards else "No awards listed"}

TASK:
Generate a comprehensive CV optimization strategy using ALL the information above. Return your response as a JSON object with this structure:

{{
  "template_structure": {{
    "format": "recommended format (chronological/functional/hybrid)",
    "sections": ["ordered list of sections - include ALL relevant sections from CV"],
    "section_priorities": {{"section_name": "why it should be included and how to optimize it"}},
    "length": "recommended page length"
  }},
  "content_recommendations": {{
    "summary": "Re-write professional summary optimized for {target_job} in {target_location} (2-3 powerful sentences highlighting most relevant experience)",
    "key_skills": ["Top 10-12 skills from the CV that are most relevant for {target_job}"],
    "experience_tips": ["Specific tips for reframing ACTUAL work experience for this role - reference specific companies/projects from CV"],
    "missing_skills": ["Skills commonly required for {target_job} that are missing from current skillset"],
    "keywords": ["15-20 ATS keywords for {target_job} - include both from CV and additional ones needed"]
  }},
  "location_specific": {{
    "format_preferences": "CV format preferences in {target_location}",
    "cultural_considerations": ["Important cultural/regional considerations for {target_location}"],
    "common_requirements": ["What employers in {target_location} typically expect"]
  }},
  "industry_insights": {{
    "trends": "Current trends in {industry or 'the target industry'}",
    "sought_after_skills": ["Top 8-10 skills employers are actively looking for"],
    "red_flags": ["Things to avoid in CV for this industry"]
  }},
  "action_items": {{
    "immediate": ["5-7 specific changes to make RIGHT NOW - reference actual CV content"],
    "important": ["7-10 important improvements - be specific about what to change"],
    "nice_to_have": ["5 optional enhancements"]
  }},
  "template_example": {{
    "professional_summary": "Example summary using ACTUAL experience from CV tailored to the target role",
    "key_achievements": ["3 example achievement bullets from ACTUAL CV formatted with metrics"],
    "skills_presentation": "How to present the ACTUAL skills from CV in optimal order"
  }}
}}

CRITICAL REQUIREMENTS:
1. Use ALL the actual information from the CV provided above
2. Reference specific companies, projects, technologies, and achievements from the CV
3. Don't invent or assume information not in the CV
4. Provide specific, actionable recommendations
5. Ensure all recommendations are optimized for both ATS systems and human recruiters
6. Use actual data points and metrics from the experience section
7. Include ALL relevant sections from the original CV in your recommendations
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