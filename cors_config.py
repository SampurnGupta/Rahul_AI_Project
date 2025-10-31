"""
CORS Configuration Guide for PDF Chatbot API

This file provides secure CORS configurations for different deployment scenarios.
"""

# DEVELOPMENT CONFIGURATION (Permissive for testing)
CORS_DEV = {
    "allow_origins": ["*"],
    "allow_credentials": False,
    "allow_methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["*"],
}

# PRODUCTION CONFIGURATION (Secure)
CORS_PROD = {
    "allow_origins": [
        "https://yourdomain.com",
        "https://www.yourdomain.com",
        "https://your-frontend-app.vercel.app",
        # Add your actual frontend domains here
    ],
    "allow_credentials": False,
    "allow_methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
    ],
}

# RENDER DEPLOYMENT CONFIGURATION
CORS_RENDER = {
    "allow_origins": [
        "https://your-service-name.onrender.com",
        "https://localhost:3000",  # For local frontend development
        "http://localhost:3000",   # For local frontend development
        # Add your actual frontend domains
    ],
    "allow_credentials": False,
    "allow_methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": [
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
    ],
}

# HACKRX COMPETITION CONFIGURATION (Optimized for competition judging)
CORS_HACKRX = {
    "allow_origins": ["*"],  # Allow all origins for judges to test from anywhere
    "allow_credentials": False,
    "allow_methods": ["GET", "POST", "OPTIONS", "HEAD"],
    "allow_headers": [
        "Accept",
        "Accept-Language",
        "Content-Language", 
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Referer",
        "User-Agent",
    ],
    "expose_headers": [
        "Content-Length",
        "Content-Type", 
        "Date",
        "Server",
    ],
    "max_age": 3600,  # Cache preflight requests for 1 hour
}

# EXTERNAL REMOTE TESTING CONFIGURATION (Optimized for external testers)
CORS_EXTERNAL_TESTING = {
    "allow_origins": ["*"],  # Allow all origins for external testing
    "allow_credentials": False,
    "allow_methods": ["GET", "POST", "OPTIONS", "HEAD"],
    "allow_headers": [
        "*",  # Allow all headers for maximum compatibility
    ],
    "expose_headers": [
        "Content-Length",
        "Content-Type",
        "Date",
        "Server",
    ],
    "max_age": 3600,  # Cache preflight requests for 1 hour
}
