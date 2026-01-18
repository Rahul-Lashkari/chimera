# Installation Guide

This guide covers how to install CHIMERA for different use cases.

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **API Keys**: Google AI API key (for Gemini models)

## Quick Installation

### Using pip (Recommended for Users)

```bash
pip install chimera-benchmark
```

### From Source (Recommended for Contributors)

```bash
# Clone the repository
git clone https://github.com/Rahul-Lashkari/chimera.git
cd chimera

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Configuration

After installation, create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Verify Installation

```bash
# Check version
chimera --version

# Run tests
pytest tests/ -v
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Configuration Reference](../configuration.md)
- [API Documentation](../api/models.md)
