#!/usr/bin/env python3
"""
ADGM Corporate Agent Startup Script
This script handles the complete startup process with error checking and logging.
"""

import sys
import os
import subprocess
import importlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adgm_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ADGMAgentLauncher:
    def __init__(self):
        self.required_packages = [
            'gradio',
            'docx',
            'requests',
            'bs4',
            'lxml',
            'pathlib2'
        ]
        
        self.optional_packages = [
            'openai',
            'sentence_transformers', 
            'sklearn',
            'numpy'
        ]
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python 3.8+ required. Current version: {version.major}.{version.minor}")
            return False
        
        logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_package_installation(self, package_name):
        """Check if a package is installed"""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    def install_missing_packages(self):
        """Install missing required packages"""
        missing_packages = []
        
        # Check required packages
        for package in self.required_packages:
            if not self.check_package_installation(package):
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing required packages: {missing_packages}")
            logger.info("Installing missing packages...")
            
            # Install from requirements.txt if it exists
            if Path("requirements.txt").exists():
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                    logger.info("✅ Successfully installed packages from requirements.txt")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install packages: {e}")
                    return False
            else:
                # Install individual packages
                for package in missing_packages:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                        logger.info(f"✅ Installed {package}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to install {package}: {e}")
                        return False
        
        return True
    
    def check_optional_packages(self):
        """Check optional packages and warn if missing"""
        missing_optional = []
        
        for package in self.optional_packages:
            if not self.check_package_installation(package):
                missing_optional.append(package)
        
        if missing_optional:
            logger.warning(f"Optional packages missing (RAG functionality may be limited): {missing_optional}")
            logger.info("To install optional packages: pip install openai sentence-transformers scikit-learn numpy")
    
    def create_sample_env_file(self):
        """Create sample .env file if it doesn't exist"""
        env_file = Path(".env")
        if not env_file.exists():
            sample_env = """# ADGM Corporate Agent Configuration
# Uncomment and add your API keys for enhanced functionality

# OpenAI API Key (for enhanced RAG)
# OPENAI_API_KEY=your-openai-api-key-here

# Anthropic Claude API Key
# ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Google Gemini API Key
# GOOGLE_API_KEY=your-google-api-key-here

# Application Settings
GRADIO_DEBUG=false
GRADIO_SHARE=true
"""
            with open(env_file, 'w') as f:
                f.write(sample_env)
            logger.info("📄 Created sample .env file")
    
    def run_application(self):
        """Run the main application"""
        try:
            logger.info("🚀 Starting ADGM Corporate Agent...")
            
            # Import and run the main application
            from app import interface
            
            logger.info("🌐 Launching web interface...")
            logger.info("📱 Interface will be available at: http://localhost:7860")
            
            # Launch with configuration
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True,
                debug=False,
                show_error=True,
                inbrowser=True,
                quiet=False
            )
            
        except ImportError as e:
            logger.error(f"Failed to import application: {e}")
            logger.error("Please ensure app.py exists and all dependencies are installed")
            return False
        except Exception as e:
            logger.error(f"Application startup failed: {e}")
            return False
        
        return True
    
    def startup_checks(self):
        """Run all startup checks"""
        logger.info("🔍 Running startup checks...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Install missing packages
        if not self.install_missing_packages():
            logger.error("❌ Failed to install required packages")
            return False
        
        # Check optional packages
        self.check_optional_packages()
        
        # Create sample env file
        self.create_sample_env_file()
        
        # Verify app.py exists
        if not Path("app.py").exists():
            logger.error("❌ app.py not found. Please ensure the main application file exists.")
            return False
        
        logger.info("✅ All startup checks passed!")
        return True
    
    def launch(self):
        """Main launch function"""
        print("🏛️ ADGM Corporate Agent - Legal Document Intelligence")
        print("=" * 60)
        
        if not self.startup_checks():
            logger.error("❌ Startup checks failed. Please fix the issues above.")
            return False
        
        # Run the application
        return self.run_application()

def main():
    """Main entry point"""
    launcher = ADGMAgentLauncher()
    
    try:
        success = launcher.launch()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()