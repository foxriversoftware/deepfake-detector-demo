import gradio as gr
from transformers import pipeline

# Load Hugging Face model
pipe = pipeline("audio-classification", model="mo-thecreator/Deepfake-audio-detection")

# Risk mapping
def detect_deepfake(audio_file):
    results = pipe(audio_file)
    fake_score = [r['score'] for r in results if r['label'].lower() == "fake"][0]

    if fake_score < 0.3:
        risk = "Low Risk"
    elif fake_score < 0.7:
        risk = "Suspicious"
    else:
        risk = "Likely Deepfake"

    return {
        "Assessment": risk,
        "Fake probability": round(fake_score, 4),
        "Raw model output": results
    }

# Gradio UI
demo = gr.Interface(
    fn=detect_deepfake,
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio"),
    outputs="json",
    title="Deepfake Audio Detector",
    description="Upload or record audio to check if it's real or fake."
)

if __name__ == "__main__":
    demo.launch()