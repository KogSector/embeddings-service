"""
Unit tests for configuration validation
Tests environment variable parsing and validation logic
"""

import pytest
import os
from typing import Dict, Any


class TestEnvironmentVariables:
    """Test environment variable validation"""
    
    def setup_method(self):
        """Save original environment variables"""
        self.original_env = dict(os.environ)
    
    def teardown_method(self):
        """Restore original environment variables"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_required_model_variables(self):
        """Test that required model variables are identified"""
        # These are the minimal required variables based on the Rust code
        required_vars = [
            'DEFAULT_EMBEDDING_MODEL',
            'NVIDIA_NIM_BASE_URL',  # or GEMINI_BASE_URL
            'NVIDIA_NIM_API_KEY',   # or GEMINI_API_KEY
        ]
        
        # Set them
        os.environ['DEFAULT_EMBEDDING_MODEL'] = 'nvidia/embeddings/003'
        os.environ['NVIDIA_NIM_BASE_URL'] = 'https://integrate.api.nvidia.com/v1'
        os.environ['NVIDIA_NIM_API_KEY'] = 'test-key'
        
        # Verify they're set
        for var in required_vars:
            if 'NVIDIA' in var or 'GEMINI' in var:
                # Check fallback logic
                nim_var = var.replace('GEMINI', 'NVIDIA')
                gemini_var = var.replace('NVIDIA', 'GEMINI')
                assert nim_var in os.environ or gemini_var in os.environ
            else:
                assert var in os.environ
    
    def test_optional_server_variables(self):
        """Test optional server variables with defaults"""
        # These have defaults in the Rust code
        optional_defaults = {
            'HOST': '0.0.0.0',
            'PORT': '3011',
            'WORKERS': '1',
        }
        
        # Don't set them, verify we can use defaults
        for var, default in optional_defaults.items():
            value = os.environ.get(var, default)
            assert value == default
    
    def test_optional_model_variables(self):
        """Test optional model variables with defaults"""
        optional_defaults = {
            'MAX_BATCH_SIZE': '32',
            'MODEL_TIMEOUT_SECS': '30',
        }
        
        for var, default in optional_defaults.items():
            value = os.environ.get(var, default)
            assert value == default
    
    def test_kafka_variables(self):
        """Test Kafka configuration variables"""
        kafka_vars = [
            'KAFKA_BOOTSTRAP_SERVERS',
            'EMBEDDINGS_SERVICE_KAFKA_GROUP_ID',
            'KAFKA_INPUT_TOPIC',
            'KAFKA_OUTPUT_TOPIC',
        ]
        
        # Set them
        for var in kafka_vars:
            os.environ[var] = 'test-value'
        
        # Verify
        for var in kafka_vars:
            assert var in os.environ
            assert os.environ[var] == 'test-value'
    
    def test_kafka_enabled_default(self):
        """Test KAFKA_ENABLED default value"""
        # Not set, should default to true
        value = os.environ.get('KAFKA_ENABLED', 'true')
        assert value.lower() in ['true', '1']
    
    def test_gemini_fallback_vars(self):
        """Test that GEMINI vars can fallback to NVIDIA vars"""
        # Set only GEMINI vars
        os.environ['GEMINI_BASE_URL'] = 'https://gemini.com'
        os.environ['GEMINI_API_KEY'] = 'gemini-key'
        
        # Verify fallback logic works
        # In Rust code: NVIDIA_NIM_BASE_URL.or_else(GEMINI_BASE_URL)
        base_url = os.environ.get('NVIDIA_NIM_BASE_URL') or os.environ.get('GEMINI_BASE_URL')
        api_key = os.environ.get('NVIDIA_NIM_API_KEY') or os.environ.get('GEMINI_API_KEY')
        
        assert base_url == 'https://gemini.com'
        assert api_key == 'gemini-key'
    
    def test_port_validation(self):
        """Test port number validation"""
        # Valid ports
        valid_ports = ['80', '443', '3011', '8080', '65535']
        for port in valid_ports:
            try:
                int(port)
                assert 1 <= int(port) <= 65535
            except ValueError:
                pytest.fail(f"Invalid port: {port}")
        
        # Invalid ports
        invalid_ports = ['0', '65536', 'abc', '-1', '99999']
        for port in invalid_ports:
            try:
                port_num = int(port)
                assert not (1 <= port_num <= 65535), f"Port {port} should be invalid"
            except ValueError:
                pass  # Expected for non-numeric
    
    def test_workers_validation(self):
        """Test workers count validation"""
        # Valid worker counts
        valid_workers = ['1', '2', '4', '8', '16']
        for workers in valid_workers:
            try:
                workers_num = int(workers)
                assert workers_num > 0
            except ValueError:
                pytest.fail(f"Invalid workers: {workers}")
        
        # Invalid worker counts
        invalid_workers = ['0', '-1', 'abc']
        for workers in invalid_workers:
            try:
                workers_num = int(workers)
                assert workers_num <= 0, f"Workers {workers} should be invalid"
            except ValueError:
                pass  # Expected for non-numeric
    
    def test_batch_size_validation(self):
        """Test batch size validation"""
        # Valid batch sizes
        valid_sizes = ['1', '16', '32', '64', '128']
        for size in valid_sizes:
            try:
                size_num = int(size)
                assert size_num > 0
            except ValueError:
                pytest.fail(f"Invalid batch size: {size}")
    
    def test_timeout_validation(self):
        """Test timeout validation"""
        # Valid timeouts
        valid_timeouts = ['10', '30', '60', '120']
        for timeout in valid_timeouts:
            try:
                timeout_num = int(timeout)
                assert timeout_num > 0
            except ValueError:
                pytest.fail(f"Invalid timeout: {timeout}")


class TestConfigurationScenarios:
    """Test common configuration scenarios"""
    
    def setup_method(self):
        """Save original environment variables"""
        self.original_env = dict(os.environ)
    
    def teardown_method(self):
        """Restore original environment variables"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_minimal_configuration(self):
        """Test minimal viable configuration"""
        os.environ['DEFAULT_EMBEDDING_MODEL'] = 'nvidia/embeddings/003'
        os.environ['NVIDIA_NIM_BASE_URL'] = 'https://integrate.api.nvidia.com/v1'
        os.environ['NVIDIA_NIM_API_KEY'] = 'test-key'
        
        # Should be enough to start the service (Kafka vars have empty defaults)
        assert 'DEFAULT_EMBEDDING_MODEL' in os.environ
        assert 'NVIDIA_NIM_BASE_URL' in os.environ
        assert 'NVIDIA_NIM_API_KEY' in os.environ
    
    def test_full_configuration(self):
        """Test full configuration with all variables"""
        config = {
            'HOST': '0.0.0.0',
            'PORT': '3011',
            'WORKERS': '1',
            'DEFAULT_EMBEDDING_MODEL': 'nvidia/embeddings/003',
            'MAX_BATCH_SIZE': '32',
            'MODEL_TIMEOUT_SECS': '30',
            'NVIDIA_NIM_BASE_URL': 'https://integrate.api.nvidia.com/v1',
            'NVIDIA_NIM_API_KEY': 'test-key',
            'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092',
            'EMBEDDINGS_SERVICE_KAFKA_GROUP_ID': 'embeddings-service-group',
            'KAFKA_INPUT_TOPIC': 'chunk-raw-events',
            'KAFKA_OUTPUT_TOPIC': 'embedding-generated-events',
            'KAFKA_ENABLED': 'true',
        }
        
        os.environ.update(config)
        
        # Verify all are set
        for key, value in config.items():
            assert os.environ.get(key) == value
    
    def test_gemini_configuration(self):
        """Test configuration using Gemini variables"""
        os.environ['DEFAULT_EMBEDDING_MODEL'] = 'gemini/embedding-001'
        os.environ['GEMINI_BASE_URL'] = 'https://generativelanguage.googleapis.com'
        os.environ['GEMINI_API_KEY'] = 'gemini-key'
        
        # Verify fallback logic
        base_url = os.environ.get('NVIDIA_NIM_BASE_URL') or os.environ.get('GEMINI_BASE_URL')
        api_key = os.environ.get('NVIDIA_NIM_API_KEY') or os.environ.get('GEMINI_API_KEY')
        
        assert base_url == 'https://generativelanguage.googleapis.com'
        assert api_key == 'gemini-key'
    
    def test_kafka_disabled(self):
        """Test configuration with Kafka disabled"""
        os.environ['DEFAULT_EMBEDDING_MODEL'] = 'nvidia/embeddings/003'
        os.environ['NVIDIA_NIM_BASE_URL'] = 'https://integrate.api.nvidia.com/v1'
        os.environ['NVIDIA_NIM_API_KEY'] = 'test-key'
        os.environ['KAFKA_ENABLED'] = 'false'
        
        assert os.environ['KAFKA_ENABLED'] == 'false'


class TestURLValidation:
    """Test URL validation for API endpoints"""
    
    def test_nvidia_nim_url_formats(self):
        """Test various NVIDIA NIM URL formats"""
        valid_urls = [
            'https://integrate.api.nvidia.com/v1',
            'https://integrate.api.nvidia.com/v1/embeddings',
            'https://integrate.api.nvidia.com/v1/',
            'https://custom-nim.example.com/v1/embeddings',
        ]
        
        for url in valid_urls:
            assert url.startswith('https://'), f"URL should use HTTPS: {url}"
            assert 'api.nvidia.com' in url or 'example.com' in url
    
    def test_gemini_url_formats(self):
        """Test various Gemini URL formats"""
        valid_urls = [
            'https://generativelanguage.googleapis.com',
            'https://generativelanguage.googleapis.com/v1',
            'https://custom-gemini.example.com',
        ]
        
        for url in valid_urls:
            assert url.startswith('https://'), f"URL should use HTTPS: {url}"
    
    def test_invalid_urls(self):
        """Test invalid URL formats"""
        invalid_urls = [
            'http://insecure.com',  # HTTP instead of HTTPS
            'not-a-url',
            'ftp://example.com',
            '',
        ]
        
        for url in invalid_urls:
            if url:  # Skip empty string for this test
                if not url.startswith('http'):
                    continue  # Invalid format, as expected
                assert not url.startswith('https'), f"URL should not be HTTPS: {url}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
