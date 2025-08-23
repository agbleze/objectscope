from tensorboard import program
from objectscope import logger
import subprocess
from PIL import Image
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

def launch_tensorboard(logdir, port_num=None):
    if not port_num:
        port_num = "default"
    tb = program.TensorBoard()
    argv = [None, "--logdir", logdir, "--port", port_num]
    tb.configure(argv)
    url = tb.launch()
    logger.info(f"TensorBoard launched at {url}")
    
    
def run_optimize_model(model_name_or_path, output_dir, device="cpu",
                       provider="CPUExecutionProvider",
                       precision="int4",
                       ):
    logger.info("Optimizing model...")
    cmd = ["olive", "auto-opt",
            "--model_name_or_path", model_name_or_path,
            "--trust_remote_code",
            "--output_path", output_dir,
            "--device", device,
            "--provider", provider,
            "--use_ort_genai",
            "--precision", precision,
            "--log_level", 1
            ]
    subprocess.run(cmd, check=True)
    logger.info("Model optimization completed.")


def compute_statistics(img_paths: list):
    channel_sum    = np.zeros(3, dtype=np.float64)
    channel_sqsum  = np.zeros(3, dtype=np.float64)
    total_pixels   = 0

    for path in img_paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float64) / 255.0 

        h, w, _ = arr.shape
        pixels = h * w

        channel_sum   += arr.sum(axis=(0, 1))
        channel_sqsum += (arr ** 2).sum(axis=(0, 1))
        total_pixels  += pixels

    mean = channel_sum / total_pixels
    var  = channel_sqsum / total_pixels - mean ** 2
    std  = np.sqrt(var)

    return {
        "chan_mean": mean,  
        "chan_std":  std,   
        "chan_var":  var,   
    }


def predict_bbox(image, model_path):
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    output = ort_session.run(None, {input_name: image})
    return {"bbox": output[0],
            "class": output[1],
            "score": output[2],
            "shape": output[3],
            }
    