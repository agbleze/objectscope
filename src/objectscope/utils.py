from tensorboard import program
from objectscope import logger
import subprocess

def launch_tensorboard(logdir, port=None):
    if not port:
        port = "default"
    tb = program.TensorBoard()
    argv = [None, "--logdir", logdir, "--port", port]
    tb.configure(argv)
    url = tb.lauch()
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
