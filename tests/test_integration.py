"""
Integration tests for embeddings-service API
Tests the deployed service at emb.confuse.site
"""

import pytest
import requests
from typing import Dict, Any
import time


class TestEmbeddingsServiceHealth:
    """Test health check endpoint"""
    
    BASE_URL = "https://emb.confuse.site"
    
    def test_health_check(self):
        """Test that the health check endpoint returns OK"""
        response = requests.get(f"{self.BASE_URL}/health", timeout=10)
        
        assert response.status_code == 200
        assert response.text == "OK"
    
    def test_root_endpoint(self):
        """Test that the root endpoint also returns OK"""
        response = requests.get(f"{self.BASE_URL}/", timeout=10)
        
        assert response.status_code == 200
        assert response.text == "OK"
    
    def test_service_responsive(self):
        """Test that the service responds within reasonable time"""
        start_time = time.time()
        response = requests.get(f"{self.BASE_URL}/health", timeout=10)
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response.status_code == 200
        assert response_time < 5.0, f"Service took {response_time:.2f}s, expected < 5s"


class TestEmbeddingsServiceConnectivity:
    """Test service connectivity and availability"""
    
    BASE_URL = "https://emb.confuse.site"
    
    def test_dns_resolution(self):
        """Test that the domain resolves correctly"""
        import socket
        try:
            socket.gethostbyname("emb.confuse.site")
            assert True
        except socket.gaierror:
            pytest.fail("Could not resolve emb.confuse.site")
    
    def test_https_connection(self):
        """Test that HTTPS connection works"""
        response = requests.get(f"{self.BASE_URL}/health", timeout=10)
        assert response.url.startswith("https://")
    
    def test_multiple_requests(self):
        """Test that service handles multiple consecutive requests"""
        for i in range(5):
            response = requests.get(f"{self.BASE_URL}/health", timeout=10)
            assert response.status_code == 200, f"Request {i+1} failed"


class TestEmbeddingsServiceHeaders:
    """Test response headers and metadata"""
    
    BASE_URL = "https://emb.confuse.site"
    
    def test_content_type(self):
        """Test that response has appropriate content type"""
        response = requests.get(f"{self.BASE_URL}/health", timeout=10)
        
        content_type = response.headers.get('Content-Type', '')
        assert 'text/plain' in content_type or 'text' in content_type.lower()
    
    def test_connection_header(self):
        """Test connection handling"""
        response = requests.get(f"{self.BASE_URL}/health", timeout=10)
        
        # Check that connection is handled properly
        connection = response.headers.get('Connection', '').lower()
        assert connection in ['', 'close', 'keep-alive']


class TestEmbeddingsServiceErrorHandling:
    """Test error handling and edge cases"""
    
    BASE_URL = "https://emb.confuse.site"
    
    def test_invalid_endpoint(self):
        """Test that invalid endpoints are handled gracefully"""
        response = requests.get(f"{self.BASE_URL}/invalid", timeout=10)
        
        # The health check server returns 200 OK for all requests
        # This is expected behavior for the simple health check implementation
        assert response.status_code == 200
        assert response.text == "OK"
    
    def test_post_to_health(self):
        """Test that POST to health endpoint is handled"""
        response = requests.post(f"{self.BASE_URL}/health", timeout=10)
        
        # Service may accept POST or return error, either is acceptable
        # as long as it doesn't crash
        assert response.status_code in [200, 405, 404]
    
    def test_large_headers(self):
        """Test that service handles requests with large headers"""
        headers = {
            'X-Custom-Header': 'x' * 1000
        }
        response = requests.get(f"{self.BASE_URL}/health", headers=headers, timeout=10)
        
        # Should handle gracefully
        assert response.status_code in [200, 431]


class TestEmbeddingsServicePerformance:
    """Test service performance characteristics"""
    
    BASE_URL = "https://emb.confuse.site"
    
    def test_response_time_consistency(self):
        """Test that response times are consistent"""
        times = []
        for _ in range(10):
            start = time.time()
            response = requests.get(f"{self.BASE_URL}/health", timeout=10)
            end = time.time()
            times.append(end - start)
            assert response.status_code == 200
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        assert avg_time < 3.0, f"Average response time {avg_time:.2f}s too high"
        assert max_time < 5.0, f"Max response time {max_time:.2f}s too high"
    
    def test_concurrent_requests(self):
        """Test that service handles concurrent requests"""
        import concurrent.futures
        
        def make_request():
            response = requests.get(f"{self.BASE_URL}/health", timeout=10)
            return response.status_code
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(r == 200 for r in results), f"Some requests failed: {results}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
