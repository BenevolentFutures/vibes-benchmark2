import gradio as gr
import random
import time
import os
import requests
import json
from dotenv import load_dotenv
import threading
from queue import Queue, Empty
import shutil

# Load environment variables
load_dotenv()

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

# Copy testquestions.txt to static directory if it exists
if os.path.exists('testquestions.txt'):
    shutil.copy2('testquestions.txt', 'static/testquestions.txt')

MAX_QUESTIONS = 10  # Maximum number of questions to support

######
# Models configuration
# 
MODELS = [
    # Standard Language Models
    {"display_name": "Claude 3 Opus", "model_id": "anthropic/claude-3-opus-20240229"},
    {"display_name": "Claude 3.5 Sonnet", "model_id": "anthropic/claude-3.5-sonnet"},
    {"display_name": "Gemini Flash 2.0 ", "model_id": "google/gemini-2.0-flash-exp:free"},
    {"display_name": "Mistral Large", "model_id": "mistralai/mistral-large-2411"},
    # {"display_name": "Claude 2.1", "model_id": "anthropic/claude-2.1"},
    {"display_name": "GPT-4o", "model_id": "openai/gpt-4o-2024-11-20"},
    # {"display_name": "GPT-3.5 Turbo", "model_id": "openai/gpt-3.5-turbo"},
    # Reasoning-specialized Models
    {"display_name": "Reasoner: O1-Mini", "model_id": "openai/o1-mini"},
    {"display_name": "Reasoner: O1 Preview", "model_id": "openai/o1-preview"},
    {"display_name": "Reasoner: DeepSeek R1", "model_id": "deepseek/deepseek-r1"},
    {"display_name": "Reasoner: Google Gemni 2.0 Flash Thinking", "model_id": "google/gemini-2.0-flash-thinking-exp:free"}
]

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
        "model": model,  # model is now the direct model_id
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
    gr.Markdown("# Vibesmark Test Suite")
    
    # Store current state
    state = gr.State({
        "questions": [], 
        "current_index": 0,
        "preferences": {},  # Store preferences for each question
        "current_model_order": {},  # Track which model is shown on which side
        "test_started": False  # Track if test has started
    })
    
    # Move model selection to the top
    with gr.Row():
        with gr.Column():
            model1_selector = gr.Dropdown(
                choices={model["model_id"]: model["display_name"] for model in MODELS},
                label="Select First Model",
                value="anthropic/claude-3.5-sonnet",
                type="value",
                allow_custom_value=False
            )
        with gr.Column():
            model2_selector = gr.Dropdown(
                choices={model["model_id"]: model["display_name"] for model in MODELS},
                label="Select Second Model",
                value="google/gemini-pro",
                type="value",
                allow_custom_value=False
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("Upload a `.txt` file with **one question per line**.")
            file_input = gr.File(label="Upload your questions (.txt)")
        with gr.Column():
            gr.Markdown("Download example questions:")
            gr.HTML('<a href="testquestions.txt" download>Download testquestions.txt</a>')
    
    with gr.Row():
        start_btn = gr.Button("Start Test", variant="primary")
        finish_btn = gr.Button("Finish & Show Results", variant="secondary", interactive=False)
        results_display = gr.Markdown("Click 'Finish & Show Results' when you're done to see the summary", visible=True)

    # Add confirmation dialog
    with gr.Row(visible=False) as confirm_dialog:
        gr.Markdown("Are you sure you want to finish the test? This will reset all progress.")
        with gr.Row():
            confirm_btn = gr.Button("Yes, Finish Test", variant="primary")
            cancel_btn = gr.Button("Cancel", variant="secondary")

    with gr.Group(visible=False) as question_group:
        question_display = gr.Markdown("### Upload a file to begin")
        with gr.Row():
            with gr.Column():
                response1_display = gr.Textbox(label="Response A", interactive=False, lines=8)
            with gr.Column():
                response2_display = gr.Textbox(label="Response B", interactive=False, lines=8)
        
        # Add preference selection buttons
        with gr.Row():
            prefer_a_btn = gr.Button("Prefer Response A", interactive=False, variant="secondary")
            preference_display = gr.Markdown("Make your selection", container=True)
            prefer_b_btn = gr.Button("Prefer Response B", interactive=False, variant="secondary")
            
        # Add vertical spacing
        gr.Row(height=30)
            
        # Move navigation to bottom of question group
        with gr.Row():
            prev_btn = gr.Button("← Previous", interactive=False)
            question_counter = gr.Markdown("Question 0 / 0")
            next_btn = gr.Button("Next →", interactive=False)

    def start_test(state, model_1, model_2):
        """Start the test and lock model selection"""
        if not state["questions"]:
            raise gr.Error("Please upload a file first.")
            
        if model_1 == model_2:
            raise gr.Error("Please select different models for comparison.")
            
        new_state = state.copy()
        new_state["test_started"] = True
        current_index = state["current_index"]
        current_question = state["questions"][current_index]
        
        # Get existing preference if any
        current_pref = state["preferences"].get(current_index, None)
        pref_display = "Make your selection"
        if current_pref is not None:
            pref_display = f"You preferred Response {current_pref}"
        
        # First yield the initial state updates
        yield [
            new_state,
            gr.update(interactive=False),  # model1_selector
            gr.update(interactive=False),  # model2_selector
            gr.update(interactive=False),  # start_btn
            gr.update(interactive=True),   # finish_btn
            "",  # response1_display
            "",  # response2_display
            gr.update(interactive=True),  # prefer_a_btn - Enable immediately
            gr.update(interactive=True),  # prefer_b_btn - Enable immediately
            pref_display,  # preference_display
            gr.update(visible=True)  # question_group
        ]
        
        # Randomly decide which model goes on which side
        if random.choice([True, False]):
            model_a, model_b = model_1, model_2
        else:
            model_a, model_b = model_2, model_1
            
        # Store the model order in state
        new_state["current_model_order"][current_index] = {
            "A": model_a,
            "B": model_b
        }

        # Stream both model responses in parallel
        for partial1, partial2 in get_responses_in_parallel(current_question, model_a, model_b):
            # Check current preference again in case it changed during streaming
            current_pref = new_state["preferences"].get(current_index, None)
            pref_display = "Make your selection"
            if current_pref is not None:
                pref_display = f"You preferred Response {current_pref}"
                
            yield [
                new_state,
                gr.update(interactive=False),  # model1_selector
                gr.update(interactive=False),  # model2_selector
                gr.update(interactive=False),  # start_btn
                gr.update(interactive=True),   # finish_btn
                partial1,  # response1_display
                partial2,  # response2_display
                gr.update(interactive=True),  # prefer_a_btn - Keep enabled during streaming
                gr.update(interactive=True),  # prefer_b_btn - Keep enabled during streaming
                pref_display,  # preference_display - Maintain current preference
                gr.update(visible=True)  # question_group
            ]

    def process_file(file, state):
        if file is None:
            raise gr.Error("Please upload a file first.")
        questions = read_questions(file)
        new_state = {
            "questions": questions, 
            "current_index": 0,
            "preferences": {},
            "current_model_order": {},
            "test_started": False
        }
        
        # Return outputs in order matching the outputs list in the event handler
        return [
            f"### Question 1:\n{questions[0]}",  # question_display
            f"Question 1 / {len(questions)}",    # question_counter
            gr.update(interactive=False),        # prev_btn
            gr.update(interactive=len(questions) > 1),  # next_btn
            gr.update(value=""),                # response1_display
            gr.update(value=""),                # response2_display
            gr.update(interactive=False),       # prefer_a_btn
            gr.update(interactive=False),       # prefer_b_btn
            "Make your selection",              # preference_display
            new_state,                          # state
            gr.update(interactive=True),        # start_btn
            gr.update(interactive=False),       # finish_btn
            gr.update(visible=False)            # question_group
        ]

    def navigate_question(direction, state, model_1, model_2):
        """Navigate to next/prev question and start fetching responses"""
        if not state["test_started"]:
            raise gr.Error("Please start the test first")

        questions = state["questions"]
        current_index = state["current_index"]
        
        if direction == "next" and current_index < len(questions) - 1:
            current_index += 1
        elif direction == "prev" and current_index > 0:
            current_index -= 1
        else:
            raise gr.Error("No more questions in that direction")
            
        new_state = state.copy()
        new_state["current_index"] = current_index
        
        # Get existing preference for this question if any
        current_pref = state["preferences"].get(current_index, None)
        pref_display = "Make your selection"
        if current_pref is not None:
            pref_display = f"You preferred Response {current_pref}"
        
        # First yield to update the question display and clear responses
        yield [
            f"### Question {current_index + 1}:\n{questions[current_index]}",  # question_display
            f"Question {current_index + 1} / {len(questions)}",               # question_counter
            gr.update(interactive=current_index > 0),                         # prev_btn
            gr.update(interactive=current_index < len(questions) - 1),        # next_btn
            "",                                                              # response1_display
            "",                                                              # response2_display
            gr.update(interactive=True),                                     # prefer_a_btn - Enable immediately
            gr.update(interactive=True),                                     # prefer_b_btn - Enable immediately
            pref_display,                                                    # preference_display
            new_state,                                                       # state
            gr.update(visible=True)                                         # question_group
        ]

        # Now start fetching responses
        current_question = questions[current_index]
        
        # Randomly decide which model goes on which side
        if random.choice([True, False]):
            model_a, model_b = model_1, model_2
        else:
            model_a, model_b = model_2, model_1
            
        # Store the model order in state
        new_state["current_model_order"][current_index] = {
            "A": model_a,
            "B": model_b
        }

        # Stream both model responses in parallel
        for partial1, partial2 in get_responses_in_parallel(current_question, model_a, model_b):
            # Check current preference again in case it changed during streaming
            current_pref = new_state["preferences"].get(current_index, None)
            pref_display = "Make your selection"
            if current_pref is not None:
                pref_display = f"You preferred Response {current_pref}"
                
            yield [
                f"### Question {current_index + 1}:\n{questions[current_index]}",  # question_display
                f"Question {current_index + 1} / {len(questions)}",               # question_counter
                gr.update(interactive=current_index > 0),                         # prev_btn
                gr.update(interactive=current_index < len(questions) - 1),        # next_btn
                partial1,                                                        # response1_display
                partial2,                                                        # response2_display
                gr.update(interactive=True),                                     # prefer_a_btn - Keep enabled during streaming
                gr.update(interactive=True),                                     # prefer_b_btn - Keep enabled during streaming
                pref_display,                                                    # preference_display - Maintain current preference
                new_state,                                                       # state
                gr.update(visible=True)                                         # question_group
            ]

    def record_preference(choice, state):
        """Record user's preference for the current question"""
        current_index = state["current_index"]
        new_state = state.copy()
        new_state["preferences"][current_index] = choice
        
        # Get the actual models for this choice
        model_order = state["current_model_order"].get(current_index, {})
        model_a = model_order.get("A", "Unknown")
        model_b = model_order.get("B", "Unknown")
        
        # Create a more detailed preference message
        if choice == "A":
            preferred_model = model_a
            other_model = model_b
        else:
            preferred_model = model_b
            other_model = model_a
            
        message = f"You preferred {preferred_model} over {other_model}"
        
        return [
            new_state,
            message
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

    def reset_interface():
        """Reset all interface elements to their initial state"""
        return [
            gr.update(interactive=True),  # model1_selector
            gr.update(interactive=True),  # model2_selector
            gr.update(interactive=True),  # start_btn
            gr.update(interactive=False), # finish_btn
            gr.update(value=""),         # response1_display
            gr.update(value=""),         # response2_display
            gr.update(interactive=False), # prefer_a_btn
            gr.update(interactive=False), # prefer_b_btn
            "Make your selection",        # preference_display
            gr.update(value="### Upload a file to begin"), # question_display
            gr.update(value="Question 0 / 0"),            # question_counter
            gr.update(interactive=False),                  # prev_btn
            gr.update(interactive=False),                  # next_btn
            {   # Fresh state
                "questions": [], 
                "current_index": 0,
                "preferences": {},
                "current_model_order": {},
                "test_started": False
            },
            gr.update(visible=False)  # question_group
        ]

    def generate_results_summary(state):
        """Generate a summary of which model was preferred for which questions"""
        if not state["preferences"]:
            return ["No preferences recorded yet."] + reset_interface()
            
        # Create a mapping of model to preferred question numbers
        model_preferences = {}
        
        for q_idx, choice in state["preferences"].items():
            # Get the model order for this question
            model_order = state["current_model_order"].get(q_idx, {})
            if not model_order:
                continue
                
            # Determine which model was preferred
            preferred_model = model_order["A"] if choice == "A" else model_order["B"]
            
            # Get display name for the model
            display_name = next((m["display_name"] for m in MODELS if m["model_id"] == preferred_model), preferred_model)
            
            if display_name not in model_preferences:
                model_preferences[display_name] = []
            model_preferences[display_name].append(str(q_idx + 1))  # +1 for 1-based indexing
            
        # Format the results
        summary_parts = []
        for model, questions in model_preferences.items():
            summary_parts.append(f"**{model}** won questions {', '.join(questions)}")
            
        summary = "### Results Summary\n" + "\n\n".join(summary_parts)
        
        # Return summary and reset interface
        return [summary] + reset_interface() + [gr.update(visible=False)]  # Hide question_group

    def show_confirm_dialog(state):
        """Show confirmation dialog if test has started"""
        if not state["test_started"] or not state["questions"]:
            return [
                gr.update(visible=False),  # confirm_dialog
                ["No test in progress to finish."] + reset_interface() + [gr.update(visible=False)]  # results and reset
            ]
        return [
            gr.update(visible=True),  # confirm_dialog
            None  # No results update
        ]

    def hide_confirm_dialog():
        """Hide the confirmation dialog"""
        return gr.update(visible=False)

    # Connect events
    file_input.change(
        fn=process_file,
        inputs=[file_input, state],
        outputs=[
            question_display,
            question_counter,
            prev_btn,
            next_btn,
            response1_display,
            response2_display,
            prefer_a_btn,
            prefer_b_btn,
            preference_display,
            state,
            start_btn,
            finish_btn,
            question_group
        ]
    )

    prev_btn.click(
        fn=navigate_question,
        inputs=[
            gr.State("prev"),
            state, 
            model1_selector, 
            model2_selector
        ],
        outputs=[
            question_display,
            question_counter,
            prev_btn,
            next_btn,
            response1_display,
            response2_display,
            prefer_a_btn,
            prefer_b_btn,
            preference_display,
            state,
            question_group
        ]
    )

    next_btn.click(
        fn=navigate_question,
        inputs=[
            gr.State("next"),
            state, 
            model1_selector, 
            model2_selector
        ],
        outputs=[
            question_display,
            question_counter,
            prev_btn,
            next_btn,
            response1_display,
            response2_display,
            prefer_a_btn,
            prefer_b_btn,
            preference_display,
            state,
            question_group
        ]
    )

    start_btn.click(
        fn=start_test,
        inputs=[state, model1_selector, model2_selector],
        outputs=[
            state,
            model1_selector,
            model2_selector,
            start_btn,
            finish_btn,
            response1_display,
            response2_display,
            prefer_a_btn,
            prefer_b_btn,
            preference_display,
            question_group
        ]
    )
    
    # Connect preference buttons
    prefer_a_btn.click(
        fn=lambda state: record_preference("A", state),
        inputs=[state],
        outputs=[state, preference_display]
    )
    
    prefer_b_btn.click(
        fn=lambda state: record_preference("B", state),
        inputs=[state],
        outputs=[state, preference_display]
    )

    # Connect results button to show confirmation first
    finish_btn.click(
        fn=show_confirm_dialog,
        inputs=[state],
        outputs=[
            confirm_dialog,
            results_display
        ]
    )

    # Connect cancel button
    cancel_btn.click(
        fn=hide_confirm_dialog,
        outputs=[confirm_dialog]
    )

    # Connect confirm button to actual finish action
    confirm_btn.click(
        fn=generate_results_summary,
        inputs=[state],
        outputs=[
            results_display,
            model1_selector,
            model2_selector,
            start_btn,
            finish_btn,
            response1_display,
            response2_display,
            prefer_a_btn,
            prefer_b_btn,
            preference_display,
            question_display,
            question_counter,
            prev_btn,
            next_btn,
            state
        ]
    ).then(
        fn=hide_confirm_dialog,
        outputs=[confirm_dialog]
    )

    # Add footer with subtle styling
    gr.Markdown("<p style='color: #666; font-size: 0.8em; text-align: center; margin-top: 2em;'>Homegrown software from the Chateau</p>")

# Enable queue for partial outputs to appear as they are yielded
demo.queue()

# Launch with the appropriate host setting for deployment
if __name__ == "__main__":
    print("\nStarting Vibesmark Test Suite...")
    print("You can access the app at: http://localhost:7860")
    
    # Create a FastAPI app to serve the example file
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/testquestions.txt")
    async def get_example_file():
        return FileResponse("testquestions.txt")
    
    # Mount FastAPI app to Gradio
    demo.app.mount("/", app)
    
    demo.launch(
        server_name="0.0.0.0",  # Allows external connections
        server_port=7860,
        share=False
    )
