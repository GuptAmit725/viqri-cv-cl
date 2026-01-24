# Viqri CV Backend - FastAPI

Modern, high-performance FastAPI backend for CV upload and parsing.

## 🚀 Why FastAPI?

- ⚡ **Faster**: Built on ASGI, async support out of the box
- 📚 **Auto Documentation**: Interactive API docs at `/docs`
- ✅ **Type Safety**: Built-in data validation with Pydantic
- 🔥 **Modern**: Python 3.8+ with type hints
- 🎯 **Better Performance**: Outperforms Flask in benchmarks

## Features

- **File Upload**: Accept PDF, DOC, and DOCX files
- **Text Extraction**: Parse documents using multiple parsing strategies
- **Information Extraction**: Extract structured CV data including:
  - Personal Information (name, email, phone, LinkedIn, GitHub)
  - Education history
  - Work Experience
  - Skills (Technical, Tools, Languages)
  - Projects
  - Certifications
  - Languages
  - Professional Summary
- **Interactive Docs**: Swagger UI at `/docs` and ReDoc at `/redoc`
- **Health Check**: Monitor API status

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or with `--break-system-packages` flag if needed:

```bash
pip install -r requirements.txt --break-system-packages
```

### 2. Run the Server

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### 3. Access Interactive Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Root
```
GET /
```

Returns API information and available endpoints.

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Viqri CV Backend is running",
  "timestamp": "2025-01-23T16:00:00"
}
```

### Upload CV
```
POST /api/upload
Content-Type: multipart/form-data
```

**Request:**
- `file`: CV file (PDF, DOC, or DOCX)

**Response:**
```json
{
  "success": true,
  "message": "CV processed successfully",
  "data": {
    "personal_info": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "+1-234-567-8900",
      "location": "New York, NY",
      "linkedin": "linkedin.com/in/johndoe",
      "github": "github.com/johndoe"
    },
    "education": [...],
    "experience": [...],
    "skills": {...},
    "projects": [...],
    "certifications": [...],
    "languages": [...],
    "summary": "...",
    "metadata": {
      "original_filename": "cv.pdf",
      "file_size": 245678,
      "file_type": "pdf",
      "upload_timestamp": "2025-01-23T16:00:00",
      "processed": true
    }
  }
}
```

### Test Extraction (Debug)
```
POST /api/test-extraction
Content-Type: multipart/form-data
```

Returns raw extracted text for debugging purposes.

### Get Statistics
```
GET /api/stats
```

Returns upload statistics and configuration.

### Cleanup Uploads
```
DELETE /api/cleanup
```

Deletes all uploaded files (use with caution).

## Project Structure

```
backend/
├── app.py                  # Main FastAPI application
├── requirements.txt        # Python dependencies
├── uploads/               # Uploaded files storage
└── parsers/              # Parser modules
    ├── __init__.py
    ├── pdf_parser.py      # PDF parsing logic
    ├── docx_parser.py     # DOCX parsing logic
    └── cv_extractor.py    # CV information extraction
```

## File Size Limits

- Maximum file size: **10MB**
- Supported formats: **PDF, DOC, DOCX**

## Error Handling

FastAPI automatically returns detailed error responses:

- `400`: Bad request (invalid file type, no file provided)
- `413`: File too large
- `422`: Validation error
- `500`: Internal server error

## Interactive API Testing

### Using Swagger UI (Recommended)

1. Open http://localhost:8000/docs
2. Click on `/api/upload` endpoint
3. Click "Try it out"
4. Upload your CV file
5. Click "Execute"
6. See the response!

### Using curl

```bash
curl -X POST -F "file=@/path/to/cv.pdf" http://localhost:8000/api/upload
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/upload"
files = {"file": open("cv.pdf", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Usage with Frontend

Update your frontend script:

```javascript
const API_URL = 'http://localhost:8000';

const formData = new FormData();
formData.append('file', selectedFile);

const response = await fetch(`${API_URL}/api/upload`, {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result.data);
```

## Development

### Hot Reload

FastAPI automatically reloads when you make changes to the code (when using `--reload` flag).

### Debug Mode

Set environment variable:
```bash
export DEBUG=True  # Linux/Mac
set DEBUG=True     # Windows
```

### Logging

FastAPI logs all requests automatically. Check terminal output for:
- Request methods and paths
- Response status codes
- Error tracebacks

## Production Deployment

### Using Gunicorn + Uvicorn Workers

```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Advantages Over Flask

1. **Performance**: 2-3x faster than Flask
2. **Async Support**: Native async/await
3. **Auto Docs**: Built-in Swagger UI
4. **Type Safety**: Request/response validation
5. **Modern**: Python 3.8+ features
6. **WebSocket Support**: Out of the box
7. **Dependency Injection**: Built-in DI system

## Future Enhancements

- [ ] Add ML-based entity extraction
- [ ] Implement caching with Redis
- [ ] Add authentication (OAuth2/JWT)
- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] Batch processing support
- [ ] WebSocket for real-time updates
- [ ] Advanced NLP for better extraction
- [ ] Rate limiting
- [ ] File format conversion
- [ ] Multi-language support

## Notes

- Uploaded files are stored in the `uploads/` directory
- Files are renamed with timestamps to avoid conflicts
- The extraction uses regex patterns and may need tuning for specific CV formats
- For production, consider adding authentication and rate limiting
- CORS is enabled for all origins (restrict in production)

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt --break-system-packages
```

### Port already in use
Change port in `app.py` or use:
```bash
uvicorn app:app --port 8001
```

### CORS issues
CORS is already configured. Ensure backend is running and accessible.

## Performance Tips

1. Use async functions for I/O operations
2. Enable caching for repeated requests
3. Use connection pooling for databases
4. Implement rate limiting
5. Use CDN for static files

---

**Happy Coding!** 🚀

For issues or questions, check the API docs at `/docs` or terminal logs.