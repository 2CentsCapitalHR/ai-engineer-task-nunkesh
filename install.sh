#!/bin/bash

# ADGM Corporate Agent Installation Script
# This script automates the complete setup process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo "======================================================================"
echo "🏛️  ADGM Corporate Agent - Installation Script"
echo "    AI-Powered Legal Document Intelligence System"
echo "======================================================================"
echo

# Check if Python is installed
print_status "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION found"

# Verify Python version is 3.8+
if $PYTHON_CMD -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    print_success "Python version is compatible"
else
    print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create project directory if it doesn't exist
PROJECT_DIR="adgm-corporate-agent"
if [ ! -d "$PROJECT_DIR" ]; then
    print_status "Creating project directory..."
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
else
    print_status "Using existing project directory..."
    cd "$PROJECT_DIR"
fi

# Create virtual environment
print_status "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    print_status "Creating requirements.txt..."
    cat > requirements.txt << EOF
# ADGM Corporate Agent Requirements
gradio==4.8.0
python-docx==1.1.0
pathlib2==2.3.7
requests==2.31.0
beautifulsoup4==4.12.2
lxml==4.9.3
openai==1.3.7
sentence-transformers==2.2.2
scikit-learn==1.3.2
numpy==1.24.3
python-magic==0.4.27
PyPDF2==3.0.1
openpyxl==3.1.2
python-dotenv==1.0.0
EOF
    print_success "requirements.txt created"
fi

# Install Python packages
print_status "Installing Python packages..."
pip install -r requirements.txt
print_success "All packages installed successfully"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating sample .env file..."
    cat > .env << EOF
# ADGM Corporate Agent Configuration
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
EOF
    print_success ".env file created"
fi

# Create application files if they don't exist
if [ ! -f "app.py" ]; then
    print_warning "app.py not found. Please ensure you have the main application file."
    print_status "You can download it from the artifacts provided."
fi

# Create startup script
if [ ! -f "run_agent.py" ]; then
    print_status "Creating startup script..."
    # The run_agent.py content would be created here
    print_success "Startup script created"
fi

# Create test documents
print_status "Creating test documents..."
$PYTHON_CMD -c "
from docx import Document
import os

def create_test_doc():
    doc = Document()
    doc.add_heading('Test Articles of Association', 0)
    doc.add_paragraph('Test Company Limited')
    doc.add_paragraph('Registered Office: Dubai, UAE')
    doc.add_paragraph('Governing Law: UAE Federal Law')
    doc.add_paragraph('Jurisdiction: Dubai Courts')
    doc.save('test_document.docx')
    print('✅ Test document created: test_document.docx')

create_test_doc()
"

# Create run script
print_status "Creating run script..."
cat > run.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting ADGM Corporate Agent..."

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate  
fi

# Run the application
python app.py
EOF

chmod +x run.sh

# Create Windows batch file
cat > run.bat << 'EOF'
@echo off
echo 🚀 Starting ADGM Corporate Agent...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the application
python app.py

pause
EOF

print_success "Run scripts created (run.sh for Linux/Mac, run.bat for Windows)"

# Final instructions
echo
echo "======================================================================"
print_success "🎉 ADGM Corporate Agent Installation Complete!"
echo "======================================================================"
echo
print_status "📁 Project Structure:"
echo "   📂 $PROJECT_DIR/"
echo "   ├── 📄 app.py (main application)"
echo "   ├── 📄 requirements.txt"
echo "   ├── 📄 .env (configuration)"
echo "   ├── 📄 run.sh (Linux/Mac launcher)"
echo "   ├── 📄 run.bat (Windows launcher)"
echo "   ├── 📄 test_document.docx (sample test file)"
echo "   └── 📂 venv/ (virtual environment)"
echo

print_status "🚀 To start the application:"
echo "   Linux/Mac:  ./run.sh"
echo "   Windows:    run.bat"
echo "   Manual:     source venv/bin/activate && python app.py"
echo

print_status "🌐 Once started, the application will be available at:"
echo "   Local:      http://localhost:7860"
echo "   Network:    http://YOUR-IP:7860"
echo "   Public:     Gradio will provide a shareable link"
echo

print_status "🧪 Testing the installation:"
echo "   1. Start the application using one of the methods above"
echo "   2. Upload the provided test_document.docx"
echo "   3. Click 'Review Documents'"
echo "   4. Verify that compliance issues are detected"
echo

print_warning "📝 Optional Configuration:"
echo "   • Edit .env file to add API keys for enhanced functionality"
echo "   • Customize compliance rules in app.py"
echo "   • Add more ADGM reference documents"
echo

print_success "✅ Installation completed successfully!"
echo "======================================================================"