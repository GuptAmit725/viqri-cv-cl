"""
LinkedIn Job Scraper
Fetches job listings from LinkedIn using multiple methods with fallbacks
"""

import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import List, Dict, Optional
from urllib.parse import quote_plus
import json
import re

logger = logging.getLogger(__name__)


class LinkedInJobScraper:
    """Scrape LinkedIn job listings with multiple fallback methods"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
    def search_jobs(
        self,
        job_title: str,
        location: str,
        experience_level: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for jobs on LinkedIn
        
        Args:
            job_title: Job title to search for
            location: Job location
            experience_level: Experience level filter
            max_results: Maximum number of results (default 10)
            
        Returns:
            List of job dictionaries
        """
        logger.info("="*60)
        logger.info("🔍 LinkedIn Job Search Started")
        logger.info("="*60)
        logger.info(f"📝 Job Title: {job_title}")
        logger.info(f"📍 Location: {location}")
        logger.info(f"👔 Experience Level: {experience_level or 'Any'}")
        logger.info(f"🎯 Max Results: {max_results}")
        
        jobs = []
        
        # Try Method 1: LinkedIn Public API approach
        try:
            logger.info("🔄 Trying Method 1: LinkedIn Public API...")
            jobs = self._search_via_public_api(job_title, location, experience_level, max_results)
            if jobs:
                logger.info(f"✅ Method 1 successful: Found {len(jobs)} jobs")
                return jobs
        except Exception as e:
            logger.warning(f"⚠️  Method 1 failed: {str(e)}")
        
        # Try Method 2: Google Jobs (LinkedIn listings)
        try:
            logger.info("🔄 Trying Method 2: Google Jobs aggregator...")
            jobs = self._search_via_google_jobs(job_title, location, max_results)
            if jobs:
                logger.info(f"✅ Method 2 successful: Found {len(jobs)} jobs")
                return jobs
        except Exception as e:
            logger.warning(f"⚠️  Method 2 failed: {str(e)}")
        
        # Try Method 3: Indeed (fallback)
        try:
            logger.info("🔄 Trying Method 3: Indeed fallback...")
            jobs = self._search_via_indeed(job_title, location, max_results)
            if jobs:
                logger.info(f"✅ Method 3 successful: Found {len(jobs)} jobs")
                return jobs
        except Exception as e:
            logger.warning(f"⚠️  Method 3 failed: {str(e)}")
        
        # If all methods fail, return sample data
        logger.warning("⚠️  All scraping methods failed, returning sample data")
        return self._generate_sample_jobs(job_title, location, max_results)
    
    def _search_via_public_api(
        self,
        job_title: str,
        location: str,
        experience_level: Optional[str],
        max_results: int
    ) -> List[Dict]:
        """Search using LinkedIn's public job search URLs"""
        
        # Build LinkedIn job search URL
        encoded_title = quote_plus(job_title)
        encoded_location = quote_plus(location)
        
        # Experience level mapping
        experience_map = {
            'entry': 'f_E=2',
            'mid': 'f_E=3',
            'senior': 'f_E=4,5',
            'lead': 'f_E=5,6',
            'executive': 'f_E=6'
        }
        
        exp_filter = f"&{experience_map.get(experience_level, '')}" if experience_level else ""
        
        url = f"https://www.linkedin.com/jobs/search/?keywords={encoded_title}&location={encoded_location}{exp_filter}&sortBy=R"
        
        logger.info(f"🌐 Fetching: {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find job cards
            job_cards = soup.find_all('div', class_='base-card', limit=max_results)
            
            if not job_cards:
                # Try alternative selectors
                job_cards = soup.find_all('li', class_='jobs-search-results__list-item', limit=max_results)
            
            jobs = []
            
            for card in job_cards:
                try:
                    job = self._parse_linkedin_job_card(card)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    logger.debug(f"Failed to parse job card: {str(e)}")
                    continue
            
            return jobs
            
        except Exception as e:
            logger.error(f"LinkedIn public API error: {str(e)}")
            raise
    
    def _parse_linkedin_job_card(self, card) -> Optional[Dict]:
        """Parse a LinkedIn job card"""
        try:
            # Extract job title
            title_elem = card.find('h3', class_='base-search-card__title') or \
                        card.find('a', class_='job-card-list__title')
            title = title_elem.get_text(strip=True) if title_elem else "Job Title"
            
            # Extract company
            company_elem = card.find('h4', class_='base-search-card__subtitle') or \
                          card.find('a', class_='job-card-container__company-name')
            company = company_elem.get_text(strip=True) if company_elem else "Company"
            
            # Extract location
            location_elem = card.find('span', class_='job-search-card__location') or \
                           card.find('span', class_='job-card-container__metadata-item')
            location = location_elem.get_text(strip=True) if location_elem else "Location"
            
            # Extract job URL
            link_elem = card.find('a', class_='base-card__full-link') or \
                       card.find('a', href=re.compile(r'/jobs/view/'))
            job_url = link_elem.get('href', '') if link_elem else ""
            
            # Clean URL
            if job_url and not job_url.startswith('http'):
                job_url = f"https://www.linkedin.com{job_url}"
            
            # Extract description snippet
            desc_elem = card.find('p', class_='base-search-card__snippet')
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            # Extract posting time
            time_elem = card.find('time', class_='job-search-card__listdate')
            posted_date = time_elem.get_text(strip=True) if time_elem else "Recently"
            
            return {
                'title': title,
                'company': company,
                'location': location,
                'url': job_url or f"https://www.linkedin.com/jobs/search/?keywords={quote_plus(title)}",
                'description': description,
                'posted_date': posted_date,
                'source': 'LinkedIn'
            }
            
        except Exception as e:
            logger.debug(f"Error parsing job card: {str(e)}")
            return None
    
    def _search_via_google_jobs(
        self,
        job_title: str,
        location: str,
        max_results: int
    ) -> List[Dict]:
        """Search using Google Jobs aggregator"""
        
        query = f"{job_title} jobs in {location} site:linkedin.com"
        encoded_query = quote_plus(query)
        
        url = f"https://www.google.com/search?q={encoded_query}&ibp=htl;jobs"
        
        logger.info(f"🌐 Fetching from Google Jobs: {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            jobs = []
            job_cards = soup.find_all('div', class_='PwjeAc', limit=max_results)
            
            for card in job_cards:
                try:
                    title_elem = card.find('div', class_='BjJfJf')
                    company_elem = card.find('div', class_='vNEEBe')
                    location_elem = card.find('div', class_='Qk80Jf')
                    
                    if title_elem:
                        jobs.append({
                            'title': title_elem.get_text(strip=True),
                            'company': company_elem.get_text(strip=True) if company_elem else "Company",
                            'location': location_elem.get_text(strip=True) if location_elem else location,
                            'url': f"https://www.linkedin.com/jobs/search/?keywords={quote_plus(title_elem.get_text(strip=True))}",
                            'description': "View full job details on LinkedIn",
                            'posted_date': "Recently",
                            'source': 'Google Jobs (LinkedIn)'
                        })
                except Exception as e:
                    logger.debug(f"Failed to parse Google job card: {str(e)}")
                    continue
            
            return jobs
            
        except Exception as e:
            logger.error(f"Google Jobs error: {str(e)}")
            raise
    
    def _search_via_indeed(
        self,
        job_title: str,
        location: str,
        max_results: int
    ) -> List[Dict]:
        """Fallback to Indeed job search"""
        
        encoded_title = quote_plus(job_title)
        encoded_location = quote_plus(location)
        
        url = f"https://www.indeed.com/jobs?q={encoded_title}&l={encoded_location}"
        
        logger.info(f"🌐 Fetching from Indeed: {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            jobs = []
            # Indeed uses different class names, adjust as needed
            job_cards = soup.find_all('div', class_='job_seen_beacon', limit=max_results)
            
            for card in job_cards:
                try:
                    title_elem = card.find('h2', class_='jobTitle')
                    company_elem = card.find('span', class_='companyName')
                    location_elem = card.find('div', class_='companyLocation')
                    link_elem = card.find('a', class_='jcs-JobTitle')
                    
                    if title_elem and company_elem:
                        job_url = f"https://www.indeed.com{link_elem.get('href', '')}" if link_elem else ""
                        
                        jobs.append({
                            'title': title_elem.get_text(strip=True),
                            'company': company_elem.get_text(strip=True),
                            'location': location_elem.get_text(strip=True) if location_elem else location,
                            'url': job_url,
                            'description': "View full job details",
                            'posted_date': "Recently",
                            'source': 'Indeed'
                        })
                except Exception as e:
                    logger.debug(f"Failed to parse Indeed job card: {str(e)}")
                    continue
            
            return jobs
            
        except Exception as e:
            logger.error(f"Indeed error: {str(e)}")
            raise
    
    def _generate_sample_jobs(
        self,
        job_title: str,
        location: str,
        max_results: int
    ) -> List[Dict]:
        """Generate sample job data when scraping fails"""
        
        logger.info("📋 Generating sample job data...")
        
        companies = [
            "Google", "Microsoft", "Amazon", "Apple", "Meta",
            "Netflix", "Tesla", "Salesforce", "Adobe", "Oracle"
        ]
        
        jobs = []
        for i in range(min(max_results, len(companies))):
            jobs.append({
                'title': f"{job_title}",
                'company': companies[i],
                'location': location,
                'url': f"https://www.linkedin.com/jobs/search/?keywords={quote_plus(job_title)}&location={quote_plus(location)}",
                'description': f"Exciting opportunity for {job_title} at {companies[i]}. Apply now to join our team!",
                'posted_date': "Recently",
                'source': 'Sample Data',
                'is_sample': True
            })
        
        return jobs
    
    def get_job_details(self, job_url: str) -> Optional[Dict]:
        """
        Fetch detailed job description from job URL
        
        Args:
            job_url: URL of the job posting
            
        Returns:
            Dictionary with detailed job information
        """
        try:
            logger.info(f"📄 Fetching job details from: {job_url}")
            
            response = self.session.get(job_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract detailed description
            desc_elem = soup.find('div', class_='description__text')
            if not desc_elem:
                desc_elem = soup.find('div', class_='show-more-less-html__markup')
            
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            return {
                'full_description': description,
                'url': job_url
            }
            
        except Exception as e:
            logger.warning(f"Failed to fetch job details: {str(e)}")
            return None


def test_scraper():
    """Test the LinkedIn job scraper"""
    scraper = LinkedInJobScraper()
    
    jobs = scraper.search_jobs(
        job_title="Software Engineer",
        location="San Francisco, CA",
        experience_level="mid",
        max_results=5
    )
    
    print(f"\nFound {len(jobs)} jobs:")
    for i, job in enumerate(jobs, 1):
        print(f"\n{i}. {job['title']}")
        print(f"   Company: {job['company']}")
        print(f"   Location: {job['location']}")
        print(f"   URL: {job['url']}")
        print(f"   Source: {job['source']}")


if __name__ == "__main__":
    test_scraper()