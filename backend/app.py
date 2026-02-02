from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from datetime import datetime
import shutil
from pathlib import Path
from dotenv import load_dotenv
import logging
import traceback
import sys

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

logger.info("="*60)
logger.info("🚀 Viqri CV API Starting")
logger.info("="*60)

# Import parsers
logger.info("📦 Importing parsers...")
from parsers.pdf_parser import parse_pdf
from parsers.docx_parser import parse_docx
from parsers.cv_extractor import extract_cv_info as extract_cv_info_regex
logger.info("✅ Parsers imported successfully")

# Import CV template generator
logger.info("📦 Importing CV template generator...")
from cv_template_generator import CVTemplateGenerator
logger.info("✅ CV template generator imported")

# Import Gemini-powered CV extractor (primary)
logger.info("📦 Importing Gemini CV extractor...")
from gemini_cv_extractor import GeminiCVExtractor
logger.info("✅ Gemini CV extractor imported")

# Import LLM-powered CV extractor (fallback)
logger.info("📦 Attempting to import Groq extractor...")
try:
    from llm_cv_extractor import LLMCVExtractor
    GROQ_AVAILABLE = True
    logger.info("✅ Groq extractor available")
except Exception as e:
    GROQ_AVAILABLE = False
    logger.warning(f"⚠️  Groq extractor not available: {str(e)}")

# Import GitHub Portfolio Generator
logger.info("📦 Importing GitHub Portfolio Generator...")
try:
    from github_portfolio_generator import GitHubPortfolioGenerator
    GITHUB_PORTFOLIO_AVAILABLE = True
    logger.info("✅ GitHub Portfolio Generator available")
except Exception as e:
    GITHUB_PORTFOLIO_AVAILABLE = False
    logger.warning(f"⚠️  GitHub Portfolio Generator not available: {str(e)}")

# Import LinkedIn job search components
logger.info("📦 Importing LinkedIn job scraper...")
try:
    from linkedin_job_scraper import LinkedInJobScraper
    LINKEDIN_SCRAPER_AVAILABLE = True
    logger.info("✅ LinkedIn job scraper imported")
except Exception as e:
    LINKEDIN_SCRAPER_AVAILABLE = False
    logger.warning(f"⚠️  LinkedIn job scraper not available: {str(e)}")

logger.info("📦 Importing Groq job analyzer...")
try:
    from groq_job_analyzer import GroqJobAnalyzer
    GROQ_JOB_ANALYZER_AVAILABLE = True
    logger.info("✅ Groq job analyzer imported")
except Exception as e:
    GROQ_JOB_ANALYZER_AVAILABLE = False
    logger.warning(f"⚠️  Groq job analyzer not available: {str(e)}")

app = FastAPI(
    title="Viqri CV API",
    description="CV Upload and Parsing API with Portfolio Generator",
    version="2.0.0"
)

# ============================================================================
# CORS Configuration - UPDATED FOR PRODUCTION
# ============================================================================

# Get allowed origins from environment variable, or use default list
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")

if ALLOWED_ORIGINS_ENV:
    # Parse comma-separated origins from environment
    ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_ENV.split(",")]
    logger.info(f"🌍 CORS origins from environment: {ALLOWED_ORIGINS}")
else:
    # Default allowed origins
    ALLOWED_ORIGINS = [
        "https://guptamit725.github.io",  # GitHub Pages - UPDATE WITH YOUR USERNAME
        "https://mysanvi.in",              # Custom domain - HTTPS
        "http://mysanvi.in",               # Custom domain - HTTP
        "http://localhost:8000",       # Local development
        "http://127.0.0.1:8000",      # Local development alternative
        "http://localhost:3000",       # Alternative local port
        "https://viqri-cv-api-5u7hdc64va-uc.a.run.app"
    ]
    logger.info(f"🌍 CORS using default origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

logger.info("✅ CORS middleware configured")

# ============================================================================

# Configuration
UPLOAD_FOLDER = Path("uploads")
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Check for API keys at startup
logger.info("🔑 Checking API keys...")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GOOGLE_API_KEY:
    logger.info(f"✅ Google/Gemini API key found (length: {len(GOOGLE_API_KEY)})")
else:
    logger.warning("⚠️  No Google/Gemini API key found")

if GROQ_API_KEY:
    logger.info(f"✅ Groq API key found (length: {len(GROQ_API_KEY)})")
else:
    logger.warning("⚠️  No Groq API key found")

if not GOOGLE_API_KEY and not GROQ_API_KEY:
    logger.error("❌ No API keys found! Will use regex fallback only")
else:
    logger.info("✅ At least one API key available for extraction")

# Create upload folder if it doesn't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)


# Pydantic models for request validation
class TemplateRequest(BaseModel):
    """Request model for CV template generation"""
    cv_data: dict
    target_job: str
    target_location: str
    industry: Optional[str] = None
    experience_level: Optional[str] = None


class JobMatchRequest(BaseModel):
    """Request model for job match analysis"""
    cv_data: dict
    job_description: str


class ImprovementRequest(BaseModel):
    """Request model for CV improvement suggestions"""
    cv_data: dict
    focus_area: str = "general"


class GitHubVerifyRequest(BaseModel):
    """Request model for GitHub token verification"""
    github_token: str


class PortfolioDeployRequest(BaseModel):
    """Request model for portfolio deployment"""
    github_token: str
    github_username: str
    repo_name: Optional[str] = None
    cv_data: dict


class JobSearchRequest(BaseModel):
    """Request model for job search"""
    keywords: str
    location: str
    experience_level: Optional[str] = "mid-senior"
    limit: Optional[int] = 10


class JobAnalysisRequest(BaseModel):
    """Request model for comprehensive job analysis"""
    cv_data: Dict[str, Any]
    jobs: List[Dict[str, Any]]
    target_job: str
    target_location: str


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """Get file extension"""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

class JobMatchRequest(BaseModel):
    """Request model for job matching"""
    cv_data: dict
    job_title: str
    location: str
    experience_level: Optional[str] = None


# ============================================================================
# NEW ENDPOINT TO ADD (before if __name__ == '__main__', around line 740)
# ============================================================================

@app.post("/api/match-jobs")
async def match_jobs(request: JobMatchRequest):
    """
    Find and match top LinkedIn jobs with CV
    
    Args:
        request: JobMatchRequest with CV data, job title, location, and experience level
        
    Returns:
        Top 10 matched jobs with summaries, insights, and match scores
    """
    try:
        logger.info("="*60)
        logger.info("🎯 Job Matching Request Received")
        logger.info("="*60)
        logger.info(f"📝 Job Title: {request.job_title}")
        logger.info(f"📍 Location: {request.location}")
        logger.info(f"👔 Experience: {request.experience_level or 'Not specified'}")
        
        # Check if job matching components are available
        if not LINKEDIN_SCRAPER_AVAILABLE or not GROQ_JOB_ANALYZER_AVAILABLE:
            missing = []
            if not LINKEDIN_SCRAPER_AVAILABLE:
                missing.append("LinkedIn job scraper")
            if not GROQ_JOB_ANALYZER_AVAILABLE:
                missing.append("Groq job analyzer")
            
            raise HTTPException(
                status_code=503,
                detail=f"Job matching feature not available. Missing: {', '.join(missing)}. Please ensure linkedin_job_scraper.py and groq_job_analyzer.py are in the backend directory."
            )
        
        # Validate CV data
        if not request.cv_data:
            raise HTTPException(status_code=400, detail="CV data is required")
        
        # Step 1: Scrape LinkedIn jobs
        logger.info("🔍 Step 1: Scraping LinkedIn jobs...")
        scraper = LinkedInJobScraper()
        jobs = scraper.search_jobs(
            job_title=request.job_title,
            location=request.location,
            experience_level=request.experience_level,
            max_results=10
        )
        
        if not jobs:
            logger.warning("⚠️  No jobs found")
            return {
                "success": False,
                "message": "No jobs found for the specified criteria",
                "jobs": [],
                "insights": {
                    "match_quality": "N/A",
                    "recommendations": ["Try adjusting your search criteria"],
                    "action_items": ["Broaden your job title or location search"]
                }
            }
        
        logger.info(f"✅ Found {len(jobs)} jobs")
        
        # Step 2: Analyze and match jobs with CV
        logger.info("🤖 Step 2: Analyzing and matching jobs...")
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            logger.warning("⚠️  Groq API key not found, returning jobs without AI analysis")
            return {
                "success": True,
                "message": f"Found {len(jobs)} jobs (AI analysis unavailable)",
                "jobs": jobs,
                "insights": {
                    "match_quality": "Good",
                    "average_match_score": 70.0,
                    "recommendations": [
                        f"Found {len(jobs)} opportunities for {request.job_title}",
                        "Review each job to find the best fit"
                    ],
                    "action_items": [
                        "Apply to top matches first",
                        "Tailor your CV for each application"
                    ]
                }
            }
        
        analyzer = GroqJobAnalyzer(api_key=groq_api_key)
        result = analyzer.analyze_and_match_jobs(
            jobs=jobs,
            cv_data=request.cv_data,
            job_title=request.job_title,
            location=request.location
        )
        
        if not result.get('success'):
            logger.error(f"❌ Job matching failed: {result.get('error')}")
            # Return jobs without matching
            return {
                "success": True,
                "message": f"Found {len(jobs)} jobs (matching analysis failed)",
                "jobs": jobs,
                "insights": {
                    "match_quality": "Good",
                    "recommendations": [f"Found {len(jobs)} opportunities"],
                    "action_items": ["Review and apply to relevant positions"]
                }
            }
        
        logger.info("✅ Job matching completed successfully")
        logger.info(f"📊 Top match score: {result.get('summary', {}).get('top_match', 0)}")
        logger.info(f"📊 Average match: {result.get('summary', {}).get('average_match', 0)}")
        
        return {
            "success": True,
            "message": f"Successfully matched {len(result['jobs'])} jobs",
            "jobs": result['jobs'],
            "insights": result['insights'],
            "summary": result['summary']
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Error in job matching: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Viqri CV API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "cors_enabled": True,
        "allowed_origins": len(ALLOWED_ORIGINS),
        "features": ["cv_parsing", "template_generation", "portfolio_generator"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Viqri CV Backend is running",
        "timestamp": datetime.now().isoformat(),
        "cors_configured": True,
        "portfolio_generator": GITHUB_PORTFOLIO_AVAILABLE,
        "job_matching": {
            "linkedin_scraper": LINKEDIN_SCRAPER_AVAILABLE,
            "groq_analyzer": GROQ_JOB_ANALYZER_AVAILABLE,
            "available": LINKEDIN_SCRAPER_AVAILABLE and GROQ_JOB_ANALYZER_AVAILABLE
        }
    }


@app.post("/api/upload")
async def upload_cv(file: UploadFile = File(...)):
    """
    Handle CV file upload and parsing
    
    Args:
        file: Uploaded CV file (PDF, DOC, or DOCX)
        
    Returns:
        JSON response with extracted CV information
    """
    logger.info("="*60)
    logger.info("📤 Upload request received")
    logger.info("="*60)
    
    # Initialize cv_data to None to avoid unbound local variable error
    cv_data = None
    
    try:
        # Check if file is provided
        if not file:
            logger.error("❌ No file provided")
            raise HTTPException(status_code=400, detail="No file provided")

        logger.info(f"📄 File received: {file.filename}")
        logger.info(f"📄 Content type: {file.content_type}")

        # Check if file is selected
        if file.filename == '':
            logger.error("❌ No file selected (empty filename)")
            raise HTTPException(status_code=400, detail="No file selected")

        # Validate file type
        if not allowed_file(file.filename):
            logger.error(f"❌ Invalid file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only PDF, DOC, and DOCX are allowed"
            )
        
        logger.info("✅ File type validated")

        # Check file size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        logger.info(f"📊 File size: {file_size_mb:.2f} MB")
        
        if len(contents) > MAX_FILE_SIZE:
            logger.error(f"❌ File too large: {file_size_mb:.2f} MB")
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024)}MB limit"
            )
        
        logger.info("✅ File size validated")

        # Reset file pointer
        await file.seek(0)

        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = file.filename.replace(" ", "_")
        unique_filename = f"{timestamp}_{original_filename}"
        filepath = UPLOAD_FOLDER / unique_filename

        # Save file
        logger.info(f"💾 Saving file: {filepath}")
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("✅ File saved successfully")

        # Get file extension
        file_ext = get_file_extension(file.filename)
        logger.info(f"📝 File extension: {file_ext}")

        # Parse file based on type
        logger.info("📖 Parsing file...")
        if file_ext == 'pdf':
            logger.info("Using PDF parser...")
            raw_text = parse_pdf(str(filepath))
            print(f"EXTRACTED RAW TEXT FROM PYTHON LIBRARY: {raw_text}")
        elif file_ext in ['doc', 'docx']:
            logger.info("Using DOCX parser...")
            raw_text = parse_docx(str(filepath))
        else:
            logger.error(f"❌ Unsupported file format: {file_ext}")
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")

        logger.info(f"✅ File parsed, extracted {len(raw_text)} characters")

        # Extract CV information using multiple methods with fallback
        logger.info("🔍 Extracting CV information...")
        
        extraction_successful = False
        extraction_method = None

        # Method 1: Try Gemini first (most accurate)
        if GOOGLE_API_KEY and not extraction_successful:
            try:
                logger.info("Attempting Gemini extraction...")
                print(f"GEMINI EXTRACTION>>>>>")
                gemini_extractor = GeminiCVExtractor(api_key=GOOGLE_API_KEY)
                cv_data = gemini_extractor.extract_cv_info(raw_text)
                
                if cv_data and cv_data.get("personal_info"):
                    extraction_successful = True
                    extraction_method = "gemini"
                    logger.info("✅ Gemini extraction successful")
            except Exception as e:
                logger.warning(f"⚠️  Gemini extraction failed: {str(e)}")

        # Method 2: Try Groq LLM as fallback
        if GROQ_API_KEY and GROQ_AVAILABLE and not extraction_successful:
            try:
                logger.info("Attempting Groq LLM extraction...")
                print(f"GROQ EXTRACTION>>>>>")
                #llm_extractor = LLMCVExtractor(api_key=GROQ_API_KEY)
                gemini_extractor = GeminiCVExtractor()
                cv_data = gemini_extractor.extract_cv_info(raw_text) #llm_extractor.extract_cv_info(raw_text)
                
                if cv_data and cv_data.get("personal_info"):
                    extraction_successful = True
                    extraction_method = "groq_llm"
                    logger.info("✅ Groq LLM extraction successful")
            except Exception as e:
                logger.warning(f"⚠️  Groq LLM extraction failed: {str(e)}")

        # Method 3: Fallback to regex extraction
        if not extraction_successful:
            logger.info("Using regex extraction as fallback...")
            cv_data = extract_cv_info_regex(raw_text)
            print(f"REGEX EXTRACTION>>>>>")
            extraction_method = "regex"
            logger.info("✅ Regex extraction completed")

        logger.info(f"✅ CV extraction completed using method: {extraction_method}")
        logger.info(f"📊 Extracted sections: {list(cv_data.keys())}")

        # Clean up uploaded file
        logger.info("🧹 Cleaning up uploaded file...")
        if filepath.exists():
            filepath.unlink()
            logger.info("✅ File cleaned up")

        return {
            "success": True,
            "message": "CV parsed successfully",
            "data": cv_data,
            "extraction_method": extraction_method,
            "file_info": {
                "filename": file.filename,
                "size_mb": round(file_size_mb, 2),
                "type": file_ext
            }
        }

    except HTTPException as he:
        # Clean up file if it exists
        if cv_data is None and 'filepath' in locals() and Path(filepath).exists():
            Path(filepath).unlink()
        raise he
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and Path(filepath).exists():
            Path(filepath).unlink()
        logger.error(f"❌ Error processing CV: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cleanup")
async def cleanup_uploads():
    """Delete all uploaded files (use with caution)"""
    try:
        if UPLOAD_FOLDER.exists():
            file_count = len(list(UPLOAD_FOLDER.glob("*")))
            for file in UPLOAD_FOLDER.glob("*"):
                if file.is_file():
                    file.unlink()
            
            return {
                "success": True,
                "message": f"Deleted {file_count} files",
                "deleted_count": file_count
            }
        return {
            "success": True,
            "message": "No files to delete",
            "deleted_count": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-template")
async def generate_cv_template(request: TemplateRequest):
    """
    Generate AI-powered CV template optimized for specific job and location
    
    Args:
        request: TemplateRequest with cv_data, target_job, target_location, etc.
        
    Returns:
        Optimized CV template and recommendations
    """
    try:
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY not configured. Please set the environment variable."
            )
        
        # Initialize generator
        generator = CVTemplateGenerator(api_key=groq_api_key)
        
        # Generate template
        result = generator.generate_template(
            cv_data=request.cv_data,
            target_job=request.target_job,
            target_location=request.target_location,
            industry=request.industry,
            experience_level=request.experience_level
        )
        
        if result.get("success") == False:
            raise HTTPException(status_code=500, detail=result.get("error", "Template generation failed"))
        
        return {
            "success": True,
            "message": "CV template generated successfully",
            "data": result
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating template: {str(e)}")


@app.post("/api/job-match")
async def analyze_job_match(request: JobMatchRequest):
    """
    Analyze how well the CV matches a job description
    
    Args:
        request: JobMatchRequest with cv_data and job_description
        
    Returns:
        Match score and detailed analysis
    """
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY not configured"
            )
        
        generator = CVTemplateGenerator(api_key=groq_api_key)
        result = generator.generate_job_match_score(
            cv_data=request.cv_data,
            job_description=request.job_description
        )
        
        if result.get("success") == False:
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return {
            "success": True,
            "message": "Job match analysis completed",
            "data": result
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/improve-cv")
async def get_cv_improvements(request: ImprovementRequest):
    """
    Get AI-powered improvement suggestions for CV
    
    Args:
        request: ImprovementRequest with cv_data and focus_area
        
    Returns:
        Detailed improvement suggestions
    """
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY not configured"
            )
        
        generator = CVTemplateGenerator(api_key=groq_api_key)
        result = generator.suggest_improvements(
            cv_data=request.cv_data,
            focus_area=request.focus_area
        )
        
        if result.get("success") == False:
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return {
            "success": True,
            "message": "Improvement suggestions generated",
            "data": result
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare-extraction")
async def compare_extraction_methods(file: UploadFile = File(...)):
    """
    Compare LLM extraction vs Regex extraction for debugging/testing
    
    Args:
        file: CV file to analyze
        
    Returns:
        Both extraction results for comparison
    """
    try:
        if not file or not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file")

        # Read and save file
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

        await file.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = file.filename.replace(" ", "_")
        unique_filename = f"{timestamp}_{original_filename}"
        filepath = UPLOAD_FOLDER / unique_filename

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_ext = get_file_extension(file.filename)

        # Parse raw text
        if file_ext == 'pdf':
            raw_text = parse_pdf(str(filepath))
        elif file_ext in ['doc', 'docx']:
            raw_text = parse_docx(str(filepath))
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

        # Extract with both methods
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        results = {
            "regex_extraction": extract_cv_info_regex(raw_text),
            "llm_extraction": None,
            "raw_text_length": len(raw_text)
        }

        if groq_api_key:
            try:
                extractor = LLMCVExtractor(api_key=groq_api_key)
                results["llm_extraction"] = extractor.extract_cv_info(raw_text)
            except Exception as e:
                results["llm_error"] = str(e)
        else:
            results["llm_extraction"] = "GROQ_API_KEY not configured"

        return {
            "success": True,
            "message": "Comparison completed",
            "data": results
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PORTFOLIO GENERATOR ENDPOINTS
# ============================================================================

@app.post("/api/verify-github-token")
async def verify_github_token(request: GitHubVerifyRequest):
    """
    Verify GitHub personal access token
    
    Args:
        request: GitHubVerifyRequest with token
        
    Returns:
        Verification status and user info
    """
    try:
        if not GITHUB_PORTFOLIO_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="GitHub Portfolio Generator not available. Please ensure github_portfolio_generator.py is in the backend directory."
            )
        
        logger.info("🔐 Verifying GitHub token...")
        generator = GitHubPortfolioGenerator(github_token=request.github_token)
        result = generator.verify_token()
        
        if result.get("success"):
            logger.info(f"✅ Token verified for user: {result.get('username')}")
            return {
                "success": True,
                "message": "GitHub token verified successfully",
                "data": result
            }
        else:
            logger.error(f"❌ Token verification failed: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Token verification failed")
            }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error verifying GitHub token: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/deploy-portfolio")
async def deploy_portfolio(request: PortfolioDeployRequest):
    """
    Deploy portfolio to GitHub Pages
    
    Args:
        request: PortfolioDeployRequest with token, username, and CV data
        
    Returns:
        Deployment status and portfolio URL
    """
    try:
        logger.info("="*60)
        logger.info("🚀 Portfolio Deployment Request Received")
        logger.info("="*60)
        
        if not GITHUB_PORTFOLIO_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="GitHub Portfolio Generator not available"
            )
        
        # Validate CV data
        if not request.cv_data:
            raise HTTPException(status_code=400, detail="CV data is required")
        
        logger.info(f"👤 Username: {request.github_username}")
        logger.info(f"📦 Repository: {request.repo_name or 'default'}")
        
        # Initialize generator
        generator = GitHubPortfolioGenerator(github_token=request.github_token)
        
        # Deploy portfolio
        logger.info(f"🚀 Deploying portfolio for user: {request.github_username}")
        result = generator.deploy_portfolio(
            cv_data=request.cv_data,
            username=request.github_username,
            repo_name=request.repo_name
        )
        
        if result.get("success"):
            logger.info(f"✅ Portfolio deployed: {result.get('portfolio_url')}")
            return {
                "success": True,
                "message": result.get("message"),
                "data": {
                    "portfolio_url": result.get("portfolio_url"),
                    "repo_name": result.get("repo_name"),
                    "note": result.get("note")
                }
            }
        else:
            logger.error(f"❌ Portfolio deployment failed: {result.get('error')}")
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Portfolio deployment failed")
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deploying portfolio: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/preview-portfolio")
async def preview_portfolio(cv_data: dict = Body(...)):
    """
    Generate portfolio HTML preview without deploying
    
    Args:
        cv_data: Extracted CV data
        
    Returns:
        Generated HTML content
    """
    try:
        if not GITHUB_PORTFOLIO_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="GitHub Portfolio Generator not available"
            )
        
        logger.info("👁️  Generating portfolio preview...")
        
        # Use a dummy token since we're just generating HTML
        generator = GitHubPortfolioGenerator(github_token="dummy")
        html_content = generator.generate_portfolio_html(cv_data)
        
        logger.info(f"✅ Preview generated ({len(html_content)} characters)")
        
        return {
            "success": True,
            "message": "Portfolio preview generated",
            "html": html_content
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating portfolio preview: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# JOB SEARCH AND ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/jobs/search")
async def search_linkedin_jobs(request: JobSearchRequest):
    """
    Search for LinkedIn jobs based on keywords and location
    """
    logger.info("="*60)
    logger.info(f"🔍 Job search request: {request.keywords} in {request.location}")
    logger.info("="*60)
    
    try:
        if not LINKEDIN_SCRAPER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="LinkedIn job scraper not available. Please install dependencies: pip install httpx beautifulsoup4 lxml"
            )
        
        scraper = LinkedInJobScraper()
        jobs = await scraper.search_jobs(
            keywords=request.keywords,
            location=request.location,
            experience_level=request.experience_level,
            limit=request.limit
        )
        
        logger.info(f"✅ Found {len(jobs)} jobs")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "jobs": jobs,
                "count": len(jobs),
                "query": {
                    "keywords": request.keywords,
                    "location": request.location,
                    "experience_level": request.experience_level
                }
            }
        })
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Job search error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Job search failed: {str(e)}"
            }
        )


@app.post("/api/jobs/analyze")
async def analyze_jobs_for_cv(request: JobAnalysisRequest):
    """
    Comprehensive job analysis and CV matching using Groq AI
    """
    logger.info("="*60)
    logger.info("🤖 Starting comprehensive job analysis")
    logger.info("="*60)
    
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=500,
                detail="Groq API key not configured. Please set GROQ_API_KEY environment variable."
            )
        
        if not JOB_MATCHING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Job analyzer not available"
            )
        
        analyzer = JobMatcher(api_key=groq_api_key)
        
        logger.info(f"📊 Analyzing {len(request.jobs)} jobs")
        
        # Step 1: Summarize jobs
        logger.info("Step 1: Summarizing job requirements...")
        jobs_summary = await analyzer._summarize_jobs(request.jobs)
        
        # Step 2: Match CV to jobs
        logger.info("Step 2: Matching CV to job requirements...")
        match_analysis = await analyzer.match_cv_to_jobs(request.cv_data, jobs_summary)
        
        # Step 3: Generate comprehensive insights
        logger.info("Step 3: Generating comprehensive insights...")
        insights = await analyzer.generate_cv_insights(
            cv_data=request.cv_data,
            jobs_summary=jobs_summary,
            match_analysis=match_analysis,
            target_job=request.target_job,
            target_location=request.target_location
        )
        
        logger.info("✅ Analysis completed successfully")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "jobs_summary": jobs_summary,
                "match_analysis": match_analysis,
                "insights": insights,
                "metadata": {
                    "jobs_analyzed": len(request.jobs),
                    "target_job": request.target_job,
                    "target_location": request.target_location,
                    "analysis_date": datetime.now().isoformat()
                }
            }
        })
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Job analysis error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Job analysis failed: {str(e)}"
            }
        )


@app.post("/api/jobs/quick-match")
async def quick_job_match(
    cv_data: Dict[str, Any] = Body(...), 
    target_job: str = Body(...), 
    target_location: str = Body(...)
):
    """
    Quick end-to-end: Search jobs + Analyze + Generate insights
    This is the main endpoint used by the frontend
    """
    logger.info("="*60)
    logger.info(f"🚀 Quick job match: {target_job} in {target_location}")
    logger.info("="*60)
    
    try:
        # Check availability
        if not LINKEDIN_SCRAPER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="LinkedIn job scraper not available. Please install dependencies."
            )
        
        # Step 1: Search for jobs
        logger.info("Step 1: Searching for relevant jobs...")
        scraper = LinkedInJobScraper()
        
        experience_level = cv_data.get('experience_level') or cv_data.get('experienceLevel') or 'mid-senior'
        
        jobs = await scraper.search_jobs(
            keywords=target_job,
            location=target_location,
            experience_level=experience_level,
            limit=10
        )
        
        logger.info(f"✅ Found {len(jobs)} jobs")
        
        if len(jobs) == 0:
            return JSONResponse(content={
                "success": True,
                "data": {
                    "jobs": [],
                    "message": "No jobs found. Try different keywords or location."
                }
            })
        
        # Step 2: Analyze jobs using Groq (if available)
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key or not GROQ_ANALYZER_AVAILABLE:
            logger.warning("⚠️  Groq API not available, returning jobs without analysis")
            return JSONResponse(content={
                "success": True,
                "data": {
                    "jobs": jobs,
                    "analysis_available": False,
                    "message": "Jobs found but AI analysis not available"
                }
            })
        
        try:
            logger.info("Step 2: Analyzing jobs with AI...")
            analyzer = GroqJobAnalyzer(api_key=groq_api_key)
            
            jobs_summary = await analyzer.summarize_jobs(jobs)
            match_analysis = await analyzer.match_cv_to_jobs(cv_data, jobs_summary)
            insights = await analyzer.generate_cv_insights(
                cv_data=cv_data,
                jobs_summary=jobs_summary,
                match_analysis=match_analysis,
                target_job=target_job,
                target_location=target_location
            )
            
            logger.info("✅ Complete analysis finished")
            
            return JSONResponse(content={
                "success": True,
                "data": {
                    "jobs": jobs,
                    "jobs_summary": jobs_summary,
                    "match_analysis": match_analysis,
                    "insights": insights,
                    "analysis_available": True,
                    "metadata": {
                        "jobs_count": len(jobs),
                        "target_job": target_job,
                        "target_location": target_location,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            })
        
        except Exception as analysis_error:
            logger.error(f"Analysis failed: {str(analysis_error)}")
            return JSONResponse(content={
                "success": True,
                "data": {
                    "jobs": jobs,
                    "analysis_available": False,
                    "message": "Jobs found but analysis failed"
                }
            })
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Quick job match error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Job matching failed: {str(e)}"
            }
        )


@app.get("/api/jobs/test")
async def test_job_search():
    """Test endpoint for job search functionality"""
    logger.info("Testing job search functionality...")
    
    try:
        if not LINKEDIN_SCRAPER_AVAILABLE:
            return {
                "success": False,
                "error": "LinkedIn job scraper not available",
                "recommendation": "Install: pip install httpx beautifulsoup4 lxml"
            }
        
        scraper = LinkedInJobScraper()
        jobs = await scraper.search_jobs(
            keywords="Software Engineer",
            location="San Francisco, CA",
            limit=3
        )
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        return {
            "success": True,
            "message": "Job search is working!",
            "test_results": {
                "scraper_working": True,
                "sample_jobs_found": len(jobs),
                "groq_api_configured": bool(groq_api_key),
                "groq_analyzer_available": GROQ_ANALYZER_AVAILABLE,
                "sample_job_titles": [job.get('title', 'N/A') for job in jobs[:3]]
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "details": traceback.format_exc()
        }


# ============================================================================
# END OF JOB SEARCH ENDPOINTS
# ============================================================================


if __name__ == '__main__':
    import uvicorn
    
    print("=" * 50)
    print("🚀 Viqri CV FastAPI Server Starting...")
    print("=" * 50)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📝 Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"📊 Max file size: {MAX_FILE_SIZE / (1024 * 1024)}MB")
    print(f"🌍 CORS enabled for: {', '.join(ALLOWED_ORIGINS)}")
    print(f"🎨 Portfolio Generator: {'✅ Available' if GITHUB_PORTFOLIO_AVAILABLE else '❌ Not Available'}")
    print("=" * 50)
    print("✅ Server running on http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("📖 ReDoc: http://localhost:8000/redoc")
    print("=" * 50)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
