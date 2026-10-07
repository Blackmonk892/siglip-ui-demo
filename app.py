"""Gradio UI for the SIGLIP 2 CPU Tester."""

import gradio as gr
from PIL import Image
import pandas as pd
import config
from model import load_model, analyze_image

def process_request(image, concepts, threshold):
    """Handles the UI interaction, executes inference, and formats the table."""
    if image is None:
        raise gr.Error("Please upload or paste an image.")
        
    if not concepts or not concepts.strip():
        raise gr.Error("Please enter at least one concept.")
        
    try:
        raw_results = analyze_image(image, concepts)
        
        # Format the raw output into a clean list for the dataframe
        formatted_data = []
        for res in raw_results:
            concept = res["concept"]
            score = float(res["score"])
            match = "YES" if score >= threshold else "NO"
            formatted_data.append([concept, round(score, 3), match])
            
        return pd.DataFrame(formatted_data, columns=["Concept", "Score", "Match"])
        
    except Exception as e:
        raise gr.Error(f"Error during analysis: {str(e)}")

# Initialize the model into memory immediately when the script runs
print("Initializing application...")
load_model()

# Build the simplified Gradio interface
with gr.Blocks(title="SIGLIP 2 CPU Tester", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h2>SIGLIP 2 CPU TESTER</h2>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Image (Upload / Paste / Drop)")
            concepts_input = gr.Textbox(
                label="Concepts (comma-separated)",
                placeholder="person, hard hat, gloves, safety vest",
                lines=2
            )
            threshold_slider = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                value=config.DEFAULT_THRESHOLD, 
                step=0.01, 
                label="Threshold"
            )
            analyze_btn = gr.Button("ANALYZE", variant="primary")
            
        with gr.Column(scale=1):
            results_output = gr.Dataframe(
                headers=["Concept", "Score", "Match"],
                datatype=["str", "number", "str"],
                label="Results",
                interactive=False
            )

    # Map the UI elements to the python function
    analyze_btn.click(
        fn=process_request,
        inputs=[image_input, concepts_input, threshold_slider],
        outputs=[results_output]
    )

if __name__ == "__main__":
    print(f"Starting Gradio server on http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)