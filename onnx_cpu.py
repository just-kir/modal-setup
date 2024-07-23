from modal import Image, App, build, enter, method, web_endpoint
import os
app = App("onnx-test")

image = Image.debian_slim().pip_install(
    "optimum[cpu]",
    "onnx",
    "onnxruntime",
    "transformers",
    "torch==2.2.0+cpu",
    "-f", "https://download.pytorch.org/whl/torch_stable.html"
)

with image.imports():
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    from fastapi import Response


MODEL_DIR = "/model"


@app.cls(image=image)
class Model:
  @build()  # add another step to the image build
  def download_model_to_folder(self):
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download("philschmid/tiny-bert-sst2-distilled", local_dir=MODEL_DIR)
  @enter()
  def setup(self):
    self.pipe = ORTModelForSequenceClassification.from_pretrained(
    "philschmid/tiny-bert-sst2-distilled",
    export=True,
    provider="CPUExecutionProvider",
    )
    self.tokenizer = AutoTokenizer.from_pretrained("philschmid/tiny-bert-sst2-distilled")

  @method()
  def inference(self, prompt):
    inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
    output = self.pipe(**inputs)
    logits = output.logits.tolist()
    return logits
    
  @web_endpoint(method="POST", docs=True)
  def web_inference(self, prompt):
    return self.inference.remote(prompt)


@app.local_entrypoint()
def main():
    print("THE CHECK IS", Model().inference.remote("I'm not interested"))
