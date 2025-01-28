import gradio as gr
import random
import time
import os
import requests
import json
from dotenv import load_dotenv
import threading
from queue import Queue, Empty

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
        "HTTP-Referer": "${SPACE_ID}.hf.space" if os.getenv('SPACE_ID') else "http://localhost:7860",
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

with gr.Blocks(title="Vibesmark Test Suite") as demo:
    gr.Markdown("# Vibesmark Test Suite\nUpload a `.txt` file with **one question per line**.")
    
    # Store current state
    state = gr.State({"questions": [], "current_index": 0})
    
    file_input = gr.File(label="Upload your questions (.txt)")
    with gr.Row():
        prev_btn = gr.Button("← Previous", interactive=False)
        question_counter = gr.Markdown("Question 0 / 0")
        next_btn = gr.Button("Next →", interactive=False)
    
    with gr.Group() as question_group:
        question_display = gr.Markdown("### Upload a file to begin")
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Model 1", open=False):
                    model1_display = gr.Markdown("")
                response1_display = gr.Textbox(label="Response 1", interactive=False, lines=4)
            with gr.Column():
                with gr.Accordion("Model 2", open=False):
                    model2_display = gr.Markdown("")
                response2_display = gr.Textbox(label="Response 2", interactive=False, lines=4)
    
    run_button = gr.Button("Run Comparison", variant="primary")

    def process_file(file, state):
        if file is None:
            raise gr.Error("Please upload a file first.")
        questions = read_questions(file)
        new_state = {"questions": questions, "current_index": 0}
        
        # Return outputs in order matching the outputs list in the event handler
        return [
            f"### Question 1:\n{questions[0]}",  # question_display
            f"Question 1 / {len(questions)}",    # question_counter
            gr.update(interactive=False),        # prev_btn
            gr.update(interactive=len(questions) > 1),  # next_btn
            gr.update(value=""),                # model1_display
            gr.update(value=""),                # response1_display
            gr.update(value=""),                # model2_display
            gr.update(value=""),                # response2_display
            new_state                           # state
        ]

    def navigate_question(direction, state):
        questions = state["questions"]
        current_index = state["current_index"]
        
        if direction == "next" and current_index < len(questions) - 1:
            current_index += 1
        elif direction == "prev" and current_index > 0:
            current_index -= 1
            
        new_state = state.copy()
        new_state["current_index"] = current_index
        
        # Return outputs in order matching the outputs list in the event handler
        return [
            f"### Question {current_index + 1}:\n{questions[current_index]}",  # question_display
            f"Question {current_index + 1} / {len(questions)}",               # question_counter
            gr.update(interactive=current_index > 0),                         # prev_btn
            gr.update(interactive=current_index < len(questions) - 1),        # next_btn
            gr.update(value=""),                                             # model1_display
            gr.update(value=""),                                             # response1_display
            gr.update(value=""),                                             # model2_display
            gr.update(value=""),                                             # response2_display
            new_state                                                        # state
        ]

    def get_responses_in_parallel(question, model1, model2):
        """
        Spawn two threads to run get_response for each model in parallel,
        queuing partial responses as they arrive. Yields tuples of
        (partial_response_model1, partial_response_model2).
        """
        queue1 = Queue()
        queue2 = Queue()

        def fill_queue(q, question, model):
            for partial_response in get_response(question, model):
                q.put(partial_response)
            q.put(None)  # Sentinel indicating completion

        # Spawn threads
        t1 = threading.Thread(target=fill_queue, args=(queue1, question, model1))
        t2 = threading.Thread(target=fill_queue, args=(queue2, question, model2))
        t1.start()
        t2.start()

        # Initialize trackers
        partial1 = ""
        partial2 = ""
        done1 = False
        done2 = False

        # Keep yielding as long as at least one thread is still producing
        while not (done1 and done2):
            try:
                item1 = queue1.get(timeout=0.1)
                if item1 is None:
                    done1 = True
                else:
                    partial1 = item1
            except Empty:
                pass

            try:
                item2 = queue2.get(timeout=0.1)
                if item2 is None:
                    done2 = True
                else:
                    partial2 = item2
            except Empty:
                pass

            yield partial1, partial2

        # Join threads and finish
        t1.join()
        t2.join()

    def run_comparison(state):
        """
        Run comparison for the current question, streaming both models'
        responses in parallel.
        """
        if not state["questions"]:
            raise gr.Error("Please upload a file first.")

        current_question = state["questions"][state["current_index"]]

        # Pick two distinct models
        model_1 = random.choice(MODELS)
        remaining_models = [m for m in MODELS if m != model_1]
        model_2 = random.choice(remaining_models)

        # Initial yield to display chosen models
        yield [
            gr.update(value=f"**{model_1}**"),
            gr.update(value=""),
            gr.update(value=f"**{model_2}**"),
            gr.update(value="")
        ]

        # Now stream both model responses in parallel
        for partial1, partial2 in get_responses_in_parallel(current_question, model_1, model_2):
            yield [
                gr.update(value=f"**{model_1}**"),
                gr.update(value=partial1),
                gr.update(value=f"**{model_2}**"),
                gr.update(value=partial2)
            ]

    # Connect events
    file_input.change(
        fn=process_file,
        inputs=[file_input, state],
        outputs=[
            question_display,
            question_counter,
            prev_btn,
            next_btn,
            model1_display,
            response1_display,
            model2_display,
            response2_display,
            state
        ]
    )

    prev_btn.click(
        fn=lambda state: navigate_question("prev", state),
        inputs=[state],
        outputs=[
            question_display,
            question_counter,
            prev_btn,
            next_btn,
            model1_display,
            response1_display,
            model2_display,
            response2_display,
            state
        ]
    )

    next_btn.click(
        fn=lambda state: navigate_question("next", state),
        inputs=[state],
        outputs=[
            question_display,
            question_counter,
            prev_btn,
            next_btn,
            model1_display,
            response1_display,
            model2_display,
            response2_display,
            state
        ]
    )

    run_button.click(
        fn=run_comparison,
        inputs=[state],
        outputs=[
            model1_display,
            response1_display,
            model2_display,
            response2_display
        ]
    )

    # Add footer with subtle styling
    gr.Markdown("<p style='color: #666; font-size: 0.8em; text-align: center; margin-top: 2em;'>Homegrown software from the Chateau</p>")

# Enable queue for partial outputs to appear as they are yielded
demo.queue()

# Launch with the appropriate host setting for deployment
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
