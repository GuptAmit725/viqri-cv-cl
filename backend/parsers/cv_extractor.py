"""
CV Information Extractor
Extracts structured information from CV text using regex and pattern matching
"""

import re
from datetime import datetime


def extract_cv_info(text):
    """
    Extract structured information from CV text
    
    Args:
        text (str): Raw text extracted from CV
        
    Returns:
        dict: Structured CV information
    """
    cv_data = {
        'personal_info': extract_personal_info(text),
        'education': extract_education(text),
        'experience': extract_experience(text),
        'skills': extract_skills(text),
        'projects': extract_projects(text),
        'certifications': extract_certifications(text),
        'languages': extract_languages(text),
        'summary': extract_summary(text)
    }
    
    return cv_data


def extract_personal_info(text):
    """Extract personal information"""
    personal_info = {
        'name': None,
        'email': None,
        'phone': None,
        'location': None,
        'linkedin': None,
        'github': None,
        'website': None
    }
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        personal_info['email'] = emails[0]
    
    # Extract phone number (various formats)
    phone_patterns = [
        r'\+?1?\s*\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',  # US format
        r'\+?[0-9]{1,3}[\s.-]?[0-9]{3,4}[\s.-]?[0-9]{3,4}[\s.-]?[0-9]{3,4}',  # International
        r'\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4}'  # (123) 456-7890
    ]
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            personal_info['phone'] = phones[0].strip()
            break
    
    # Extract LinkedIn
    linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/profile/view\?id=)([A-Za-z0-9_-]+)'
    linkedin = re.search(linkedin_pattern, text, re.IGNORECASE)
    if linkedin:
        personal_info['linkedin'] = f"linkedin.com/in/{linkedin.group(1)}"
    
    # Extract GitHub
    github_pattern = r'(?:github\.com/)([A-Za-z0-9_-]+)'
    github = re.search(github_pattern, text, re.IGNORECASE)
    if github:
        personal_info['github'] = f"github.com/{github.group(1)}"
    
    # Extract website
    website_pattern = r'(?:https?://)?(?:www\.)?([A-Za-z0-9-]+\.[A-Za-z]{2,}(?:/[^\s]*)?)'
    websites = re.findall(website_pattern, text)
    for site in websites:
        if 'linkedin' not in site.lower() and 'github' not in site.lower():
            personal_info['website'] = site
            break
    
    # Extract name (usually first line or before email)
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and len(line) < 50 and not any(char.isdigit() for char in line):
            # Likely to be a name if it's short and has no numbers
            if not re.search(email_pattern, line):
                personal_info['name'] = line
                break
    
    # Extract location (city, state/country)
    location_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2}|[A-Z][a-z]+)\b'
    location = re.search(location_pattern, text)
    if location:
        personal_info['location'] = f"{location.group(1)}, {location.group(2)}"
    
    return personal_info


def extract_education(text):
    """Extract education information"""
    education = []
    
    # Common education keywords
    edu_keywords = ['education', 'academic', 'qualification', 'degree']
    
    # Find education section
    edu_section = ""
    lines = text.split('\n')
    in_edu_section = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Check if we're entering education section
        if any(keyword in line_lower for keyword in edu_keywords):
            in_edu_section = True
            continue
        
        # Check if we're leaving education section
        if in_edu_section and any(keyword in line_lower for keyword in ['experience', 'work', 'employment', 'skills', 'projects']):
            break
        
        if in_edu_section and line.strip():
            edu_section += line + "\n"
    
    # Extract degree information
    degree_patterns = [
        r'(Bachelor|Master|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.|MBA)',
        r'(B\.Tech|M\.Tech|B\.E\.|M\.E\.)',
        r'(Diploma|Certificate|Associate)'
    ]
    
    # Look for university/college names
    university_pattern = r'(?:University|College|Institute|School)\s+(?:of\s+)?([A-Za-z\s]+)'
    
    # Parse education entries
    edu_lines = edu_section.split('\n')
    current_entry = {}
    
    for line in edu_lines:
        line = line.strip()
        if not line:
            if current_entry:
                education.append(current_entry)
                current_entry = {}
            continue
        
        # Check for degree
        for pattern in degree_patterns:
            degree_match = re.search(pattern, line, re.IGNORECASE)
            if degree_match:
                current_entry['degree'] = line
                break
        
        # Check for university
        uni_match = re.search(university_pattern, line, re.IGNORECASE)
        if uni_match:
            current_entry['institution'] = line
        
        # Check for year (YYYY or YYYY-YYYY)
        year_pattern = r'(19|20)\d{2}'
        years = re.findall(year_pattern, line)
        if years:
            current_entry['year'] = ' - '.join(years) if len(years) > 1 else years[0]
        
        # Check for GPA
        gpa_pattern = r'(?:GPA|CGPA)[\s:]*([0-9]\.[0-9]{1,2})'
        gpa_match = re.search(gpa_pattern, line, re.IGNORECASE)
        if gpa_match:
            current_entry['gpa'] = gpa_match.group(1)
    
    if current_entry:
        education.append(current_entry)
    
    return education


def extract_experience(text):
    """Extract work experience"""
    experience = []
    
    # Common experience keywords
    exp_keywords = ['experience', 'employment', 'work history', 'professional experience']
    
    # Find experience section
    exp_section = ""
    lines = text.split('\n')
    in_exp_section = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if any(keyword in line_lower for keyword in exp_keywords):
            in_exp_section = True
            continue
        
        if in_exp_section and any(keyword in line_lower for keyword in ['education', 'skills', 'projects', 'certifications']):
            break
        
        if in_exp_section and line.strip():
            exp_section += line + "\n"
    
    # Extract job entries
    exp_lines = exp_section.split('\n')
    current_job = {}
    
    for line in exp_lines:
        line = line.strip()
        if not line:
            if current_job and 'title' in current_job:
                experience.append(current_job)
                current_job = {}
            continue
        
        # Check for job title (usually starts with capital letter)
        if line and line[0].isupper() and '|' not in line and '@' not in line:
            if 'title' not in current_job:
                current_job['title'] = line
            elif 'company' not in current_job:
                current_job['company'] = line
        
        # Check for dates
        date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'
        dates = re.findall(date_pattern, line, re.IGNORECASE)
        if dates:
            current_job['duration'] = ' - '.join(dates) if len(dates) > 1 else dates[0]
        
        # Check for location
        if '|' in line or '@' in line:
            parts = re.split(r'[|@]', line)
            if len(parts) >= 2:
                current_job['location'] = parts[-1].strip()
        
        # Collect responsibilities (lines starting with bullet points or dashes)
        if line.startswith(('•', '-', '–', '*', '>')):
            if 'responsibilities' not in current_job:
                current_job['responsibilities'] = []
            current_job['responsibilities'].append(line.lstrip('•-–*> '))
    
    if current_job and 'title' in current_job:
        experience.append(current_job)
    
    return experience


def extract_skills(text):
    """Extract skills"""
    skills = {
        'technical': [],
        'soft': [],
        'tools': [],
        'languages': []
    }
    
    # Common skill section keywords
    skill_keywords = ['skills', 'technical skills', 'core competencies', 'expertise']
    
    # Find skills section
    skill_section = ""
    lines = text.split('\n')
    in_skill_section = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if any(keyword in line_lower for keyword in skill_keywords):
            in_skill_section = True
            continue
        
        if in_skill_section and any(keyword in line_lower for keyword in ['experience', 'education', 'projects', 'certifications']):
            break
        
        if in_skill_section and line.strip():
            skill_section += line + "\n"
    
    # Common programming languages
    prog_languages = ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'Go', 'Rust', 'TypeScript']
    
    # Common tools and frameworks
    tools = ['React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'Git']
    
    # Extract skills using pattern matching
    skill_lines = skill_section.split('\n')
    
    for line in skill_lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for programming languages
        for lang in prog_languages:
            if lang.lower() in line.lower():
                if lang not in skills['languages']:
                    skills['languages'].append(lang)
        
        # Check for tools
        for tool in tools:
            if tool.lower() in line.lower():
                if tool not in skills['tools']:
                    skills['tools'].append(tool)
        
        # Split by common delimiters
        items = re.split(r'[,;|•]', line)
        for item in items:
            item = item.strip()
            if item and len(item) > 2 and len(item) < 30:
                # Categorize as technical if it looks technical
                if any(keyword in item.lower() for keyword in ['programming', 'development', 'software', 'system']):
                    skills['technical'].append(item)
                elif item not in skills['languages'] and item not in skills['tools']:
                    skills['technical'].append(item)
    
    # Remove duplicates
    for key in skills:
        skills[key] = list(dict.fromkeys(skills[key]))
    
    return skills


def extract_projects(text):
    """Extract projects"""
    projects = []
    
    # Find projects section
    project_keywords = ['projects', 'personal projects', 'key projects']
    
    project_section = ""
    lines = text.split('\n')
    in_project_section = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if any(keyword in line_lower for keyword in project_keywords):
            in_project_section = True
            continue
        
        if in_project_section and any(keyword in line_lower for keyword in ['experience', 'education', 'skills', 'certifications']):
            break
        
        if in_project_section and line.strip():
            project_section += line + "\n"
    
    # Parse projects
    project_lines = project_section.split('\n')
    current_project = {}
    
    for line in project_lines:
        line = line.strip()
        if not line:
            if current_project and 'name' in current_project:
                projects.append(current_project)
                current_project = {}
            continue
        
        # Project name (usually bold or starts with capital)
        if line and line[0].isupper() and 'name' not in current_project:
            current_project['name'] = line
        
        # Description (lines starting with bullet points)
        if line.startswith(('•', '-', '–', '*', '>')):
            if 'description' not in current_project:
                current_project['description'] = []
            current_project['description'].append(line.lstrip('•-–*> '))
        
        # Technologies (often in parentheses or after colon)
        tech_match = re.search(r'\((.*?)\)', line)
        if tech_match:
            current_project['technologies'] = tech_match.group(1)
    
    if current_project and 'name' in current_project:
        projects.append(current_project)
    
    return projects


def extract_certifications(text):
    """Extract certifications"""
    certifications = []
    
    cert_keywords = ['certifications', 'certificates', 'licenses']
    
    cert_section = ""
    lines = text.split('\n')
    in_cert_section = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if any(keyword in line_lower for keyword in cert_keywords):
            in_cert_section = True
            continue
        
        if in_cert_section and any(keyword in line_lower for keyword in ['experience', 'education', 'skills', 'projects']):
            break
        
        if in_cert_section and line.strip():
            cert_section += line + "\n"
    
    # Extract certification entries
    cert_lines = cert_section.split('\n')
    
    for line in cert_lines:
        line = line.strip()
        if line and len(line) > 5:
            cert_entry = {'name': line}
            
            # Extract year if present
            year_pattern = r'(19|20)\d{2}'
            years = re.findall(year_pattern, line)
            if years:
                cert_entry['year'] = years[0]
            
            certifications.append(cert_entry)
    
    return certifications


def extract_languages(text):
    """Extract spoken languages"""
    languages = []
    
    lang_keywords = ['languages', 'language proficiency']
    
    # Common languages
    common_languages = ['English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese', 'Hindi', 'Arabic', 'Portuguese', 'Russian']
    
    # Check entire text for language mentions
    for lang in common_languages:
        pattern = rf'\b{lang}\b'
        if re.search(pattern, text, re.IGNORECASE):
            # Check for proficiency level
            proficiency_pattern = rf'{lang}\s*[:\-]?\s*(Native|Fluent|Professional|Intermediate|Basic|Elementary)'
            proficiency_match = re.search(proficiency_pattern, text, re.IGNORECASE)
            
            if proficiency_match:
                languages.append({
                    'language': lang,
                    'proficiency': proficiency_match.group(1)
                })
            else:
                languages.append({
                    'language': lang,
                    'proficiency': 'Unknown'
                })
    
    return languages


def extract_summary(text):
    """Extract professional summary"""
    summary_keywords = ['summary', 'profile', 'objective', 'about me', 'professional summary']
    
    lines = text.split('\n')
    in_summary = False
    summary_text = ""
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Start of summary section
        if any(keyword in line_lower for keyword in summary_keywords):
            in_summary = True
            continue
        
        # End of summary section
        if in_summary and any(keyword in line_lower for keyword in ['experience', 'education', 'skills', 'projects']):
            break
        
        # Collect summary text
        if in_summary and line.strip():
            summary_text += line + " "
        
        # Stop after collecting reasonable amount
        if in_summary and len(summary_text) > 500:
            break
    
    return summary_text.strip() if summary_text else None