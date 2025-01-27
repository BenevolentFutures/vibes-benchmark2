import gradio as gr
import random
import time

MAX_QUESTIONS = 10  # Maximum number of questions to support

######
# Fix the models
# 
MODELS = [
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "google/gemini-pro",
    "meta-llama/llama-2-70b-chat",
    "mistral/mistral-medium",
    "deepseek/deepseek-coder",
    "deepseek/deepseek-r1",
]
#
######

######
# Add OpenRouter here
# 
def get_response(question, model):
    # Simulate an API call with a random delay
    time.sleep(random.uniform(0.5, 1.5))
    return f"Sample response from {model} for: {question}"
#
######

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
        # We have 4 fields per question (model1, response1, model2, response2)
        # => total of MAX_QUESTIONS * 4 output components
        updates = [gr.update(value="")] * (MAX_QUESTIONS * 4)
        
        # Process each question, 2 models per question
        for i, question in enumerate(questions):
            # 1) Pick first model, yield it
            model_1 = random.choice(MODELS)
            updates[i*4] = gr.update(value=f"**{model_1}**")  # model1 for question i
            yield updates  # partial update (reveal model_1 accordion)
            
            # 2) Get response from model_1
            response_1 = get_response(question, model_1)
            updates[i*4 + 1] = gr.update(value=response_1)   # response1
            yield updates
            
            # 3) Pick second model (ensure different from first), yield it
            remaining_models = [m for m in MODELS if m != model_1]
            model_2 = random.choice(remaining_models)
            updates[i*4 + 2] = gr.update(value=f"**{model_2}**")  # model2
            yield updates
            
            # 4) Get response from model_2
            response_2 = get_response(question, model_2)
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
