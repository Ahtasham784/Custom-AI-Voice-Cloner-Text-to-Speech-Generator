# Custom-AI-Voice-Cloner-Text-to-Speech-Generator
pip install torch torchaudio librosa coqui-tts soundfile gradio numpy
from TTS.api import TTS
import gradio as gr

# Load the model on CPU
model_name = "tts_models/en/ljspeech/tacotron2-DDC"
tts = TTS(model_name=model_name, progress_bar=False)
tts.to("cpu")  # 👈 Ensure model runs on CPU

# Define inference function
def generate_voice(text):
    output_path = "output.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path

# Gradio interface
interface = gr.Interface(
    fn=generate_voice,
    inputs=gr.Textbox(label="Enter text to speak"),
    outputs=gr.Audio(label="Generated Voice"),
    title="AI Voice Cloner & TTS Generator",
    description="Type text and hear it spoken in the cloned voice."
)

if __name__ == "__main__":
    interface.launch()
