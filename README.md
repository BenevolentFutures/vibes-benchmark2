# Vibes Benchmark v0.1

A tool for benchmarking different AI models by comparing their responses to custom questions.

## Prerequisites

- Python 3.8 or higher
- An OpenRouter API key ([Get one here](https://openrouter.ai/))

## Setup

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd vibes-benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenRouter API key

## Usage

1. Prepare a text file with your questions (one per line)
2. Run the application:
   ```bash
   python app.py
   ```
3. Upload your questions file through the web interface
4. Click "Run Benchmark" to start comparing model responses

## Features

- Compare responses from different AI models side by side
- Supports up to 10 questions per benchmark
- Randomly selects different models for comparison
- Real-time response generation

## Supported Models

- Claude 3 Opus
- Claude 3 Sonnet
- Gemini Pro
- Mistral Medium
- Claude 2.1
- GPT-4 Turbo
- GPT-3.5 Turbo

## License

[Your chosen license]

Run it with 
`python app.py` 
