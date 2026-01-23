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

# Load environment variables from .env file
load_dotenv()

# Import parsers
from parsers.pdf_parser import parse_pdf
from parsers.docx_parser import parse_docx
from parsers.cv_extractor import extract_cv_info as extract_cv_info_regex

# Import CV template generator
from cv_template_generator import CVTemplateGenerator

# Import Gemini-powered CV extractor (primary)
from gemini_cv_extractor import GeminiCVExtractor

# Import LLM-powered CV extractor (fallback)
try:
    from llm_cv_extractor import LLMCVExtractor
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False

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
    try:
        # Check if file is provided
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check if file is selected
        if file.filename == '':
            raise HTTPException(status_code=400, detail="No file selected")

        # Validate file type
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only PDF, DOC, and DOCX are allowed"
            )

        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024)}MB limit"
            )

        # Reset file pointer
        await file.seek(0)

        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = file.filename.replace(" ", "_")
        unique_filename = f"{timestamp}_{original_filename}"
        filepath = UPLOAD_FOLDER / unique_filename

        # Save the file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get file extension
        file_ext = get_file_extension(file.filename)

        # Parse the CV based on file type
        raw_text = ""
        
        if file_ext == 'pdf':
            raw_text = parse_pdf(str(filepath))
        elif file_ext in ['doc', 'docx']:
            raw_text = parse_docx(str(filepath))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Extract structured information using AI
        # Priority: Gemini -> Groq -> Regex
        extraction_method = "regex"
        
        # Try Gemini first (best)
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_api_key:
            try:
                extractor = GeminiCVExtractor(api_key=google_api_key)
                cv_data = extractor.extract_cv_info(raw_text)
                extraction_method = "gemini"
            except Exception as e:
                print(f"Gemini extraction failed: {str(e)}")
                cv_data = None
        
        # Fallback to Groq if Gemini failed
        if not google_api_key or cv_data is None:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key and GROQ_AVAILABLE:
                try:
                    extractor = LLMCVExtractor(api_key=groq_api_key)
                    cv_data = extractor.extract_cv_info(raw_text)
                    extraction_method = "groq"
                except Exception as e:
                    print(f"Groq extraction failed: {str(e)}")
                    cv_data = None
        
        # Final fallback to regex
        if cv_data is None:
            cv_data = extract_cv_info_regex(raw_text)
            extraction_method = "regex"

        # Add metadata
        cv_data['metadata'] = {
            'original_filename': file.filename,
            'file_size': len(contents),
            'file_type': file_ext,
            'upload_timestamp': datetime.now().isoformat(),
            'processed': True,
            'extraction_method': extraction_method
        }

        # Return the extracted data
        return {
            "success": True,
            "message": "CV processed successfully",
            "data": cv_data
        }

    except HTTPException as he:
        raise he
    except Exception as e:
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