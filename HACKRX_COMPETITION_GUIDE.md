# HackRx 6.0 Competition Deployment Guide

## 🎯 API Structure Compliance

Your API now fully complies with the HackRx 6.0 competition requirements:

### ✅ **Required Endpoint**
```
POST /hackrx/run
```

### ✅ **Authentication**
```
Authorization: Bearer hackrx_2024_secret_key
```

### ✅ **Request Format**
```json
POST /hackrx/run
Content-Type: application/json
Accept: application/json
Authorization: Bearer <api_key>

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
}
```

### ✅ **Response Format**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "There is a waiting period of thirty-six (36) months..."
    ]
}
```

## 🚀 Quick Deployment to Render

### Step 1: Deploy Your Code

```bash
# Commit your changes
git add .
git commit -m "HackRx 6.0 competition-ready API"
git push origin main
```

### Step 2: Create Render Service

1. Go to https://render.com and login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Use these exact settings:

**Service Configuration:**
- **Name**: `hackrx-pdf-chatbot`
- **Environment**: `Python`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1`

**Environment Variables:**
```
ENVIRONMENT=hackrx
OPENAI_API_KEY=your_actual_openai_key
PINECONE_API_KEY=your_actual_pinecone_key
PINECONE_ENVIRONMENT=gcp-starter
HACKRX_API_KEY=hackrx_2024_secret_key
LOG_LEVEL=INFO
```

### Step 3: Test Your Deployment

Once deployed, test with the provided script:

```bash
python hackrx_test.py https://your-service.onrender.com
```

## 🎮 Competition Features

### **Enhanced for Hackathons:**

1. **Performance Monitoring**
   - Request timing and statistics
   - Cached document processing
   - Optimized error handling

2. **Comprehensive Logging**
   - Detailed request/response logging
   - Performance metrics
   - Error tracking with emojis for easy identification

3. **Robust Authentication**
   - Bearer token validation
   - Clear error messages
   - Security headers

4. **Competition-Specific Optimizations**
   - Document caching for repeated tests
   - Extended timeouts for large files
   - Performance statistics tracking

### **API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check with competition info |
| `/health` | GET | Detailed health status and stats |
| `/hackrx/run` | POST | Main competition endpoint |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |
| `/openapi.json` | GET | OpenAPI specification |

### **Request Validation:**

- ✅ Document URL format validation
- ✅ Questions list validation (1-20 questions)
- ✅ Empty string handling
- ✅ Content type validation
- ✅ Authentication validation

### **Error Handling:**

| Error | Status Code | Description |
|-------|-------------|-------------|
| Missing auth | 401 | No Bearer token provided |
| Invalid auth | 401 | Wrong API key |
| Invalid URL | 422 | Document URL validation failed |
| Empty questions | 422 | No valid questions provided |
| Download timeout | 408 | Document download timed out |
| Processing error | 500 | Internal processing error |

## 🧪 Testing Your API

### **Automated Testing**

Use the provided test script:
```bash
python hackrx_test.py https://your-service.onrender.com hackrx_2024_secret_key
```

### **Manual Testing with curl**

```bash
# Health check
curl https://your-service.onrender.com/

# Main endpoint test
curl -X POST "https://your-service.onrender.com/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer hackrx_2024_secret_key" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
  }'
```

### **Browser Testing**

Visit your deployed API documentation:
- **Swagger UI**: `https://your-service.onrender.com/docs`
- **ReDoc**: `https://your-service.onrender.com/redoc`

## 📊 Performance Considerations

### **Free Tier Optimizations:**

1. **Cold Start Handling**
   - First request may take 60+ seconds
   - Subsequent requests are much faster
   - Health check endpoint to warm up the service

2. **Document Caching**
   - Documents are cached for repeated testing
   - Reduces processing time for same documents
   - Memory-efficient caching strategy

3. **Request Limits**
   - Maximum 20 questions per request
   - Reasonable timeouts to prevent hanging
   - Graceful error handling for edge cases

### **Production Recommendations:**

1. **Upgrade to Paid Plan**
   - No cold starts
   - Better performance
   - More memory and CPU

2. **Environment Variables**
   - Use strong API keys
   - Monitor usage limits
   - Set appropriate log levels

## 🏆 Competition Readiness Checklist

- [x] ✅ Exact endpoint: `POST /hackrx/run`
- [x] ✅ Bearer token authentication
- [x] ✅ Correct request/response format
- [x] ✅ Document URL processing
- [x] ✅ Question answering functionality
- [x] ✅ Error handling and validation
- [x] ✅ API documentation
- [x] ✅ Performance monitoring
- [x] ✅ Comprehensive logging
- [x] ✅ Test automation

## 🎯 Competition Tips

1. **Test Early and Often**
   - Use the provided test script regularly
   - Test with different document types
   - Verify response format consistency

2. **Monitor Performance**
   - Check `/health` endpoint for statistics
   - Monitor response times
   - Ensure stable operation

3. **Handle Edge Cases**
   - Large documents
   - Malformed URLs
   - Network timeouts
   - Invalid questions

4. **Documentation**
   - Keep API docs updated
   - Provide clear examples
   - Include error response formats

Your API is now competition-ready for HackRx 6.0! 🎉
