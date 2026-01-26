"""
Groq Job Analyzer
Uses Groq LLM to analyze jobs, summarize requirements, and match with CV
Renamed from job_matcher.py to match app.py imports
"""

from groq import Groq
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class GroqJobAnalyzer:
    """Analyze and match jobs using Groq LLM"""
    
    def __init__(self, api_key: str):
        """
        Initialize Groq Job Analyzer
        
        Args:
            api_key: Groq API key
        """
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
    
    def analyze_and_match_jobs(
        self,
        jobs: List[Dict],
        cv_data: Dict[str, Any],
        job_title: str,
        location: str
    ) -> Dict[str, Any]:
        """
        Analyze jobs and match with CV
        
        Args:
            jobs: List of job dictionaries
            cv_data: Parsed CV data
            job_title: Target job title
            location: Target location
            
        Returns:
            Dictionary with analyzed jobs and insights
        """
        logger.info("="*60)
        logger.info("🤖 Job Analysis & Matching Started")
        logger.info("="*60)
        logger.info(f"📊 Analyzing {len(jobs)} jobs")
        
        try:
            # Step 1: Summarize each job
            logger.info("📝 Step 1: Summarizing jobs...")
            summarized_jobs = self._summarize_jobs(jobs, job_title)
            logger.info(f"✅ Summarized {len(summarized_jobs)} jobs")
            
            # Step 2: Match CV to each job
            logger.info("🎯 Step 2: Matching CV to jobs...")
            matched_jobs = self._match_cv_to_jobs(summarized_jobs, cv_data)
            logger.info(f"✅ Matched {len(matched_jobs)} jobs")
            
            # Step 3: Generate insights
            logger.info("💡 Step 3: Generating insights...")
            insights = self._generate_insights(matched_jobs, cv_data, job_title, location)
            logger.info("✅ Insights generated")
            
            # Step 4: Rank jobs by match score
            logger.info("📊 Step 4: Ranking jobs...")
            ranked_jobs = sorted(matched_jobs, key=lambda x: x.get('match_score', 0), reverse=True)
            logger.info(f"✅ Jobs ranked by relevance")
            
            return {
                'success': True,
                'jobs': ranked_jobs,
                'insights': insights,
                'summary': {
                    'total_jobs': len(ranked_jobs),
                    'average_match': sum(j.get('match_score', 0) for j in ranked_jobs) / len(ranked_jobs) if ranked_jobs else 0,
                    'top_match': ranked_jobs[0].get('match_score', 0) if ranked_jobs else 0
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in job analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'jobs': jobs,  # Return original jobs without analysis
                'insights': self._get_fallback_insights(job_title, location)
            }
    
    def _summarize_jobs(self, jobs: List[Dict], job_title: str) -> List[Dict]:
        """
        Use LLM to summarize each job's requirements
        
        Args:
            jobs: List of job dictionaries
            job_title: Target job title
            
        Returns:
            Jobs with summaries added
        """
        try:
            # Create batch prompt for efficiency
            jobs_text = "\n\n".join([
                f"Job {i+1}:\nTitle: {job['title']}\nCompany: {job['company']}\nLocation: {job['location']}"
                for i, job in enumerate(jobs[:10])
            ])
            
            prompt = f"""Analyze these {len(jobs)} job postings for "{job_title}" positions and provide a brief summary of key requirements for each.

{jobs_text}

For each job, provide:
1. A 2-3 sentence summary of the role
2. Top 3-5 key requirements/skills
3. Experience level needed

Respond in JSON format:
{{
  "jobs": [
    {{
      "job_number": 1,
      "summary": "Brief role description...",
      "key_requirements": ["Requirement 1", "Requirement 2", ...],
      "experience_level": "Mid-level"
    }},
    ...
  ]
}}"""
            
            logger.info("🚀 Calling Groq for job summaries...")
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert recruiter and job analyst. Provide concise, accurate job summaries in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            logger.info("✅ Received job summaries")
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    analysis = json.loads(json_str)
                    
                    # Add summaries to jobs
                    for i, job in enumerate(jobs[:10]):
                        if i < len(analysis.get('jobs', [])):
                            job_analysis = analysis['jobs'][i]
                            job['summary'] = job_analysis.get('summary', 'No summary available')
                            job['key_requirements'] = job_analysis.get('key_requirements', [])
                            job['experience_level'] = job_analysis.get('experience_level', 'Not specified')
                        else:
                            job['summary'] = f"Exciting opportunity at {job['company']}"
                            job['key_requirements'] = []
                            job['experience_level'] = 'Not specified'
                    
                    return jobs
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"⚠️  Failed to parse JSON, using fallback summaries: {str(e)}")
                # Add generic summaries
                for job in jobs:
                    job['summary'] = f"Join {job['company']} as a {job['title']} in {job['location']}"
                    job['key_requirements'] = []
                    job['experience_level'] = 'Not specified'
                return jobs
                
        except Exception as e:
            logger.error(f"❌ Error summarizing jobs: {str(e)}")
            # Return jobs with generic summaries
            for job in jobs:
                job['summary'] = f"Opportunity at {job['company']}"
                job['key_requirements'] = []
                job['experience_level'] = 'Not specified'
            return jobs
    
    def _match_cv_to_jobs(self, jobs: List[Dict], cv_data: Dict) -> List[Dict]:
        """
        Match CV to each job and calculate match scores
        
        Args:
            jobs: List of jobs with summaries
            cv_data: Parsed CV data
            
        Returns:
            Jobs with match scores and recommendations
        """
        try:
            # Extract CV highlights
            skills = cv_data.get('skills', {})
            all_skills = []
            for skill_list in skills.values():
                if isinstance(skill_list, list):
                    all_skills.extend(skill_list)
            
            experience = cv_data.get('experience', [])
            education = cv_data.get('education', [])
            
            cv_summary = {
                'skills': all_skills[:20],  # Top 20 skills
                'years_experience': len(experience),
                'companies': [exp.get('company', '') for exp in experience[:3]],
                'education': [edu.get('degree', '') for edu in education]
            }
            
            # Create matching prompt
            jobs_for_matching = [{
                'job_number': i+1,
                'title': job['title'],
                'company': job['company'],
                'key_requirements': job.get('key_requirements', [])
            } for i, job in enumerate(jobs[:10])]
            
            prompt = f"""Match this candidate's CV to these job opportunities and provide match scores.

Candidate Profile:
- Skills: {', '.join(cv_summary['skills'][:10])}
- Experience: {cv_summary['years_experience']} positions
- Companies: {', '.join(cv_summary['companies'])}
- Education: {', '.join(cv_summary['education'])}

Jobs:
{json.dumps(jobs_for_matching, indent=2)}

For each job, provide:
1. Match score (0-100)
2. Why it's a good fit (1-2 sentences)
3. Top 2-3 skills to highlight in application
4. Any gaps to address

Respond in JSON:
{{
  "matches": [
    {{
      "job_number": 1,
      "match_score": 85,
      "fit_reason": "Strong match because...",
      "skills_to_highlight": ["Skill 1", "Skill 2"],
      "gaps": ["Gap 1"]
    }},
    ...
  ]
}}"""
            
            logger.info("🚀 Calling Groq for CV matching...")
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert career counselor specializing in job matching. Provide accurate match scores and actionable advice in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            logger.info("✅ Received match analysis")
            
            # Parse and apply match scores
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    analysis = json.loads(json_str)
                    
                    for i, job in enumerate(jobs[:10]):
                        if i < len(analysis.get('matches', [])):
                            match = analysis['matches'][i]
                            job['match_score'] = match.get('match_score', 50)
                            job['fit_reason'] = match.get('fit_reason', '')
                            job['skills_to_highlight'] = match.get('skills_to_highlight', [])
                            job['gaps'] = match.get('gaps', [])
                        else:
                            job['match_score'] = 50
                            job['fit_reason'] = 'Potential match based on your profile'
                            job['skills_to_highlight'] = []
                            job['gaps'] = []
                    
                    return jobs
                else:
                    raise ValueError("No JSON in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"⚠️  Failed to parse matching JSON: {str(e)}")
                # Add default scores
                for i, job in enumerate(jobs):
                    job['match_score'] = 70 - (i * 3)  # Descending scores
                    job['fit_reason'] = 'Based on your experience and skills'
                    job['skills_to_highlight'] = cv_summary['skills'][:3]
                    job['gaps'] = []
                return jobs
                
        except Exception as e:
            logger.error(f"❌ Error matching CV to jobs: {str(e)}")
            # Add default scores
            for i, job in enumerate(jobs):
                job['match_score'] = 65
                job['fit_reason'] = 'Matches your profile'
                job['skills_to_highlight'] = []
                job['gaps'] = []
            return jobs
    
    def _generate_insights(
        self,
        jobs: List[Dict],
        cv_data: Dict,
        job_title: str,
        location: str
    ) -> Dict:
        """
        Generate insights and recommendations
        
        Args:
            jobs: Matched jobs
            cv_data: CV data
            job_title: Target job title
            location: Target location
            
        Returns:
            Dictionary with insights
        """
        try:
            # Calculate statistics
            match_scores = [j.get('match_score', 0) for j in jobs]
            avg_match = sum(match_scores) / len(match_scores) if match_scores else 0
            
            # Get most common requirements
            all_requirements = []
            for job in jobs:
                all_requirements.extend(job.get('key_requirements', []))
            
            # Get unique skills
            cv_skills = set()
            for skill_list in cv_data.get('skills', {}).values():
                if isinstance(skill_list, list):
                    cv_skills.update(skill_list)
            
            insights = {
                'match_quality': 'Excellent' if avg_match >= 80 else 'Good' if avg_match >= 60 else 'Fair',
                'average_match_score': round(avg_match, 1),
                'top_match_score': max(match_scores) if match_scores else 0,
                'recommendations': [
                    f"Found {len(jobs)} opportunities matching '{job_title}' in {location}",
                    f"Your profile matches {round(avg_match)}% with available positions",
                    f"Top match: {jobs[0]['title']} at {jobs[0]['company']}" if jobs else "No matches found"
                ],
                'action_items': [
                    "Apply to top 3-5 matches first",
                    "Tailor your CV for each application",
                    "Highlight relevant skills in your cover letter"
                ],
                'skills_in_demand': list(set(all_requirements[:10])) if all_requirements else [],
                'your_strengths': list(cv_skills)[:5]
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generating insights: {str(e)}")
            return self._get_fallback_insights(job_title, location)
    
    def _get_fallback_insights(self, job_title: str, location: str) -> Dict:
        """Generate fallback insights when AI analysis fails"""
        return {
            'match_quality': 'Good',
            'average_match_score': 65.0,
            'top_match_score': 75,
            'recommendations': [
                f"Found opportunities for '{job_title}' in {location}",
                "Review each job carefully to find the best fit",
                "Tailor your application to highlight relevant experience"
            ],
            'action_items': [
                "Apply to positions that closely match your skills",
                "Research each company before applying",
                "Customize your CV for each application"
            ],
            'skills_in_demand': [],
            'your_strengths': []
        }