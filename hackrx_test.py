#!/usr/bin/env python3
"""
HackRx 6.0 Competition API Test Script

This script tests the API with the exact format required for HackRx 6.0 competition.
It validates authentication, request/response format, and performance.
"""

import sys
import requests
import json
import time
from typing import Dict, List

class HackRxAPITester:
    def __init__(self, base_url: str, api_key: str = "hackrx_2024_secret_key"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'User-Agent': 'HackRx-6.0-Tester/1.0'
        })
    
    def test_health_endpoints(self) -> bool:
        """Test health check endpoints."""
        print("🏥 Testing Health Endpoints...")
        print("-" * 50)
        
        # Test root endpoint
        try:
            response = self.session.get(f"{self.base_url}/", timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint: {data.get('message', 'OK')}")
                print(f"   Version: {data.get('version', 'Unknown')}")
                print(f"   Competition: {data.get('competition', 'Unknown')}")
            else:
                print(f"❌ Root endpoint failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
        
        # Test health endpoint
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health endpoint: {data.get('status', 'Unknown')}")
                print(f"   Environment: {data.get('environment', 'Unknown')}")
                print(f"   Cached docs: {data.get('cached_documents', 0)}")
            else:
                print(f"❌ Health endpoint failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
            return False
        
        return True
    
    def test_authentication(self) -> bool:
        """Test authentication requirements."""
        print("\n🔐 Testing Authentication...")
        print("-" * 50)
        
        # Test without auth
        session_no_auth = requests.Session()
        session_no_auth.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        test_payload = {
            "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "questions": ["What is this document about?"]
        }
        
        try:
            response = session_no_auth.post(
                f"{self.base_url}/hackrx/run",
                json=test_payload,
                timeout=30
            )
            if response.status_code == 401:
                print("✅ Authentication required (401 without Bearer token)")
            else:
                print(f"⚠️  Expected 401, got {response.status_code} (auth might be disabled)")
        except Exception as e:
            print(f"❌ Auth test error: {e}")
            return False
        
        # Test with wrong auth
        session_wrong_auth = requests.Session()
        session_wrong_auth.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Bearer wrong_key'
        })
        
        try:
            response = session_wrong_auth.post(
                f"{self.base_url}/hackrx/run",
                json=test_payload,
                timeout=30
            )
            if response.status_code == 401:
                print("✅ Invalid authentication rejected (401 with wrong token)")
            else:
                print(f"⚠️  Expected 401, got {response.status_code}")
        except Exception as e:
            print(f"❌ Wrong auth test error: {e}")
            return False
        
        return True
    
    def test_competition_format(self) -> bool:
        """Test with the exact HackRx competition format."""
        print("\n🎯 Testing HackRx Competition Format...")
        print("-" * 50)
        
        # Test case matching the exact specification
        competition_payload = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
                "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                "What is the waiting period for pre-existing diseases (PED) to be covered?",
                "Does this policy cover maternity expenses, and what are the conditions?",
                "What is the waiting period for cataract surgery?",
                "Are the medical expenses for an organ donor covered under this policy?",
                "What is the No Claim Discount (NCD) offered in this policy?",
                "Is there a benefit for preventive health check-ups?",
                "How does the policy define a 'Hospital'?",
                "What is the extent of coverage for AYUSH treatments?",
                "Are there any sub-limits on room rent and ICU charges for Plan A?"
            ]
        }
        
        print(f"📄 Testing with competition document URL")
        print(f"❓ Testing with {len(competition_payload['questions'])} questions")
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/hackrx/run",
                json=competition_payload,
                timeout=300  # Extended timeout for competition document
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            print(f"⏱️  Response time: {response_time:.2f} seconds")
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response format
                if 'answers' not in data:
                    print("❌ Response missing 'answers' field")
                    return False
                
                answers = data['answers']
                questions = competition_payload['questions']
                
                if len(answers) != len(questions):
                    print(f"❌ Answer count mismatch: {len(answers)} answers for {len(questions)} questions")
                    return False
                
                print(f"✅ Competition format test passed!")
                print(f"   Questions: {len(questions)}")
                print(f"   Answers: {len(answers)}")
                print(f"   Response time: {response_time:.2f}s")
                
                # Display sample answers
                print("\n📋 Sample Answers:")
                for i, (q, a) in enumerate(zip(questions[:3], answers[:3])):
                    print(f"   Q{i+1}: {q[:80]}...")
                    print(f"   A{i+1}: {a[:120]}...")
                    print()
                
                return True
            else:
                print(f"❌ Competition test failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("⏰ Competition test timed out")
            print("   This might be normal for large documents on free hosting")
            return False
        except Exception as e:
            print(f"❌ Competition test error: {e}")
            return False
    
    def test_api_documentation(self) -> bool:
        """Test API documentation endpoints."""
        print("\n📚 Testing API Documentation...")
        print("-" * 50)
        
        # Test Swagger UI
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=30)
            if response.status_code == 200:
                print("✅ Swagger UI documentation accessible")
            else:
                print(f"❌ Swagger UI not accessible: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Swagger UI error: {e}")
            return False
        
        # Test ReDoc
        try:
            response = requests.get(f"{self.base_url}/redoc", timeout=30)
            if response.status_code == 200:
                print("✅ ReDoc documentation accessible")
            else:
                print(f"❌ ReDoc not accessible: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ ReDoc error: {e}")
            return False
        
        # Test OpenAPI spec
        try:
            response = requests.get(f"{self.base_url}/openapi.json", timeout=30)
            if response.status_code == 200:
                print("✅ OpenAPI specification accessible")
                spec = response.json()
                print(f"   Title: {spec.get('info', {}).get('title', 'Unknown')}")
                print(f"   Version: {spec.get('info', {}).get('version', 'Unknown')}")
            else:
                print(f"❌ OpenAPI spec not accessible: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ OpenAPI spec error: {e}")
            return False
        
        return True
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid inputs."""
        print("\n🚨 Testing Error Handling...")
        print("-" * 50)
        
        test_cases = [
            {
                "name": "Empty document URL",
                "payload": {"documents": "", "questions": ["Test question"]},
                "expected_status": 422
            },
            {
                "name": "Invalid document URL",
                "payload": {"documents": "not-a-url", "questions": ["Test question"]},
                "expected_status": 422
            },
            {
                "name": "Empty questions",
                "payload": {"documents": "https://example.com/test.pdf", "questions": []},
                "expected_status": 422
            },
            {
                "name": "Missing documents field",
                "payload": {"questions": ["Test question"]},
                "expected_status": 422
            },
            {
                "name": "Missing questions field",
                "payload": {"documents": "https://example.com/test.pdf"},
                "expected_status": 422
            }
        ]
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/hackrx/run",
                    json=test_case["payload"],
                    timeout=30
                )
                
                if response.status_code == test_case["expected_status"]:
                    print(f"✅ {test_case['name']}: Correctly returned {response.status_code}")
                else:
                    print(f"⚠️  {test_case['name']}: Expected {test_case['expected_status']}, got {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {test_case['name']}: Error - {e}")
                return False
        
        return True
    
    def run_comprehensive_test(self) -> bool:
        """Run comprehensive test suite for HackRx competition."""
        print("🎮 HackRx 6.0 Competition API Comprehensive Test")
        print("=" * 60)
        print(f"🌐 Testing API: {self.base_url}")
        print(f"🔑 API Key: {self.api_key}")
        print("=" * 60)
        
        tests = [
            ("Health Endpoints", self.test_health_endpoints),
            ("Authentication", self.test_authentication),
            ("API Documentation", self.test_api_documentation),
            ("Error Handling", self.test_error_handling),
            ("Competition Format", self.test_competition_format),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                    print(f"\n✅ {test_name}: PASSED")
                else:
                    print(f"\n❌ {test_name}: FAILED")
            except Exception as e:
                print(f"\n💥 {test_name}: ERROR - {e}")
        
        print("\n" + "=" * 60)
        print(f"🏁 Test Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! API is ready for HackRx 6.0 competition!")
        elif passed_tests >= total_tests - 1:
            print("✅ API is mostly ready. Minor issues may need attention.")
        else:
            print("⚠️  API has significant issues that need to be fixed.")
        
        return passed_tests == total_tests

def main():
    if len(sys.argv) < 2:
        print("Usage: python hackrx_test.py <api_base_url> [api_key]")
        print("Example: python hackrx_test.py https://your-service.onrender.com")
        print("Example: python hackrx_test.py https://your-service.onrender.com hackrx_2024_secret_key")
        sys.exit(1)
    
    base_url = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else "hackrx_2024_secret_key"
    
    tester = HackRxAPITester(base_url, api_key)
    
    try:
        success = tester.run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
