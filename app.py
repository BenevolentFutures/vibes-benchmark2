import gradio as gr
import random
import time
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MAX_QUESTIONS = 10  # Maximum number of questions to support

######
# Fix the models
# 
MODELS = [
    "anthropic/claude-3-opus-20240229",
    "anthropic/claude-3-sonnet-20240229",
    "google/gemini-pro",
    "mistralai/mistral-medium",  # Updated from mistral-7b-instruct
    "anthropic/claude-2.1",
    "openai/gpt-4-turbo-preview",
    "openai/gpt-3.5-turbo"
]
#
######

# Get configuration from environment variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL')

if not OPENROUTER_API_KEY or not OPENROUTER_BASE_URL:
    raise ValueError("Missing required environment variables. Please check your .env file.")

def get_response(question, model):
    """Get response from OpenRouter API for the given question and model."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:7860",  # Replace with your actual domain
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": question}
        ],
        "stream": True
    }
    
    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=data,
            timeout=30,  # 30 second timeout
            stream=True
        )
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    json_str = line[6:]  # Remove 'data: ' prefix
                    if json_str.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(json_str)
                        if chunk['choices'][0]['delta'].get('content'):
                            content = chunk['choices'][0]['delta']['content']
                            full_response += content
                            yield full_response
                    except json.JSONDecodeError:
                        continue
        
        return full_response
        
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to get response from {model}: {str(e)}"

def read_questions(file_obj):
    """Read questions from uploaded file and return as list"""
    with open(file_obj.name, 'r') as file:
        questions = [q.strip() for q in file.readlines() if q.strip()]
        if len(questions) > MAX_QUESTIONS:
            raise gr.Error(f"Maximum {MAX_QUESTIONS} questions allowed.")
    return questions

with gr.Blocks() as demo:
    gr.Markdown("# Vibes Benchmark\nUpload a `.txt` file with **one question per line**.")
    
    file_input = gr.File(label="Upload your questions (.txt)")
    run_button = gr.Button("Run Benchmark", variant="primary")
    
    # Create dynamic response areas
    response_areas = []
    for i in range(MAX_QUESTIONS):
        with gr.Group(visible=False) as group_i:
            gr.Markdown(f"### Question {i+1}")
            with gr.Row():
                with gr.Column():
                    # Accordion for Model 1
                    with gr.Accordion("Model 1", open=False):
                        model1_i = gr.Markdown("")
                    response1_i = gr.Textbox(label="Response 1", interactive=False, lines=4)
                with gr.Column():
                    # Accordion for Model 2
                    with gr.Accordion("Model 2", open=False):
                        model2_i = gr.Markdown("")
                    response2_i = gr.Textbox(label="Response 2", interactive=False, lines=4)
            gr.Markdown("---")
            
        response_areas.append({
            'group': group_i,
            'model1': model1_i,
            'response1': response1_i,
            'model2': model2_i,
            'response2': response2_i
        })

    def process_file(file):
        """Show/hide question groups depending on how many questions are in the file."""
        if file is None:
            raise gr.Error("Please upload a file first.")
        questions = read_questions(file)
        
        # Show as many question groups as needed; hide the rest
        updates = []
        for i in range(MAX_QUESTIONS):
            updates.append(gr.update(visible=(i < len(questions))))
        
        return updates

    def run_benchmark(file):
        """Generator function yielding partial updates in real time."""
        questions = read_questions(file)
        
        # Initialize all update values as blank
        updates = [gr.update(value="")] * (MAX_QUESTIONS * 4)
        
        # Process each question, 2 models per question
        for i, question in enumerate(questions):
            # 1) Pick first model, yield it
            model_1 = random.choice(MODELS)
            updates[i*4] = gr.update(value=f"**{model_1}**")  # model1 for question i
            yield updates  # partial update (reveal model_1 accordion)
            
            # 2) Get response from model_1
            for response_1 in get_response(question, model_1):
                updates[i*4 + 1] = gr.update(value=response_1)   # response1
                yield updates
            
            # 3) Pick second model (ensure different from first), yield it
            remaining_models = [m for m in MODELS if m != model_1]
            model_2 = random.choice(remaining_models)
            updates[i*4 + 2] = gr.update(value=f"**{model_2}**")  # model2
            yield updates
            
            # 4) Get response from model_2
            for response_2 in get_response(question, model_2):
                updates[i*4 + 3] = gr.update(value=response_2)  # response2
                yield updates

    # The outputs we update after each yield
    update_targets = []
    for area in response_areas:
        update_targets.append(area['model1'])
        update_targets.append(area['response1'])
        update_targets.append(area['model2'])
        update_targets.append(area['response2'])

    # Connect events
    file_input.change(
        fn=process_file,
        inputs=file_input,
        outputs=[area['group'] for area in response_areas]
    )

    run_button.click(
        fn=run_benchmark,
        inputs=file_input,
        outputs=update_targets
    )

# Enable queue for partial outputs to appear as they are yielded
demo.queue()
demo.launch()
