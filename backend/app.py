from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
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

app = FastAPI(
    title="Viqri CV API",
    description="CV Upload and Parsing API",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """Get file extension"""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Viqri CV API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Viqri CV Backend is running",
        "timestamp": datetime.now().isoformat()
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
        
        logger.info(f"💾 Saving file to: {filepath}")

        # Save the file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info("✅ File saved successfully")

        # Get file extension
        file_ext = get_file_extension(file.filename)
        logger.info(f"📝 File extension: {file_ext}")

        # Parse the CV based on file type
        raw_text = ""
        logger.info(f"📖 Parsing {file_ext.upper()} file...")
        
        if file_ext == 'pdf':
            raw_text = parse_pdf(str(filepath))
        elif file_ext in ['doc', 'docx']:
            raw_text = parse_docx(str(filepath))
        else:
            logger.error(f"❌ Unsupported file format: {file_ext}")
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        text_length = len(raw_text)
        logger.info(f"✅ Text extracted: {text_length} characters")
        logger.info(f"📝 First 200 chars: {raw_text[:200]}...")

        # Extract structured information using AI
        # Priority: Gemini -> Groq -> Regex
        extraction_method = "none"
        logger.info("="*60)
        logger.info("🤖 Starting CV extraction...")
        logger.info("="*60)
        
        # Try Gemini first (best)
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        logger.info(f"🔑 Checking for Gemini API key... {'Found' if google_api_key else 'Not found'}")
        
        if google_api_key:
            logger.info("🔷 Attempting Gemini extraction...")
            try:
                extractor = GeminiCVExtractor(api_key=google_api_key)
                logger.info("✅ Gemini extractor initialized")
                cv_data = extractor.extract_cv_info(raw_text)
                extraction_method = "gemini"
                logger.info("✅ Gemini extraction successful!")
                logger.info(f"📊 Extracted data keys: {list(cv_data.keys())}")
            except Exception as e:
                logger.error(f"❌ Gemini extraction failed: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                cv_data = None
        else:
            logger.warning("⚠️  No Gemini API key, skipping Gemini extraction")
        
        # Fallback to Groq if Gemini failed
        if cv_data is None:
            logger.info("🔶 Gemini failed or unavailable, trying Groq...")
            groq_api_key = os.getenv("GROQ_API_KEY")
            logger.info(f"🔑 Checking for Groq API key... {'Found' if groq_api_key else 'Not found'}")
            
            if groq_api_key and GROQ_AVAILABLE:
                logger.info("🔶 Attempting Groq extraction...")
                try:
                    extractor = LLMCVExtractor(api_key=groq_api_key)
                    logger.info("✅ Groq extractor initialized")
                    cv_data = extractor.extract_cv_info(raw_text)
                    extraction_method = "groq"
                    logger.info("✅ Groq extraction successful!")
                    logger.info(f"📊 Extracted data keys: {list(cv_data.keys())}")
                except Exception as e:
                    logger.error(f"❌ Groq extraction failed: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    cv_data = None
            else:
                logger.warning("⚠️  Groq unavailable or no API key")
        
        # Final fallback to regex
        if cv_data is None:
            logger.info("🔸 Both AI methods failed, using regex fallback...")
            try:
                cv_data = extract_cv_info_regex(raw_text)
                extraction_method = "regex"
                logger.info("✅ Regex extraction successful!")
                logger.info(f"📊 Extracted data keys: {list(cv_data.keys())}")
            except Exception as e:
                logger.error(f"❌ Even regex extraction failed: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise HTTPException(
                    status_code=500,
                    detail="All extraction methods failed. Please try a different file."
                )
        
        logger.info(f"✅ Final extraction method: {extraction_method}")
        
        # Verify cv_data is valid
        if cv_data is None:
            logger.error("❌ cv_data is None after all extraction attempts!")
            raise HTTPException(
                status_code=500,
                detail="Failed to extract CV data"
            )
        
        if not isinstance(cv_data, dict):
            logger.error(f"❌ cv_data is not a dict: {type(cv_data)}")
            raise HTTPException(
                status_code=500,
                detail="Invalid CV data format"
            )

        # Add metadata
        logger.info("📦 Adding metadata...")
        cv_data['metadata'] = {
            'original_filename': file.filename,
            'file_size': len(contents),
            'file_type': file_ext,
            'upload_timestamp': datetime.now().isoformat(),
            'processed': True,
            'extraction_method': extraction_method
        }
        
        logger.info("✅ Metadata added")
        logger.info("="*60)
        logger.info("✅ CV PROCESSING COMPLETE!")
        logger.info("="*60)

        # Return the extracted data
        response = {
            "success": True,
            "message": "CV processed successfully",
            "data": cv_data
        }
        logger.info(f"📤 Returning response with {len(str(response))} characters")
        return response

    except HTTPException as he:
        logger.error(f"❌ HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error("="*60)
        logger.error("❌ UNEXPECTED ERROR IN UPLOAD")
        logger.error("="*60)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        logger.error("="*60)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/api/test-extraction")
async def test_extraction(file: UploadFile = File(...)):
    """
    Test endpoint to see raw extracted text
    Useful for debugging
    
    Args:
        file: Uploaded CV file
        
    Returns:
        Raw extracted text and metadata
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

        await file.seek(0)

        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = file.filename.replace(" ", "_")
        unique_filename = f"{timestamp}_{original_filename}"
        filepath = UPLOAD_FOLDER / unique_filename

        # Save file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_ext = get_file_extension(file.filename)

        if file_ext == 'pdf':
            raw_text = parse_pdf(str(filepath))
        elif file_ext in ['doc', 'docx']:
            raw_text = parse_docx(str(filepath))
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

        return {
            "success": True,
            "raw_text": raw_text,
            "filename": file.filename,
            "length": len(raw_text)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get API statistics"""
    upload_count = len(list(UPLOAD_FOLDER.glob("*"))) if UPLOAD_FOLDER.exists() else 0
    
    return {
        "total_uploads": upload_count,
        "upload_folder": str(UPLOAD_FOLDER),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "allowed_extensions": list(ALLOWED_EXTENSIONS)
    }


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


if __name__ == '__main__':
    import uvicorn
    
    print("=" * 50)
    print("🚀 Viqri CV FastAPI Server Starting...")
    print("=" * 50)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📝 Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"📊 Max file size: {MAX_FILE_SIZE / (1024 * 1024)}MB")
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