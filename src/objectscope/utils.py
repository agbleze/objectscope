from tensorboard import program
from objectscope import logger


def launch_tensorboard(logdir, port=None):
    if not port:
        port = "default"
    tb = program.TensorBoard()
    argv = [None, "--logdir", logdir, "--port", port]
    tb.configure(argv)
    url = tb.lauch()
    logger.info(f"TensorBoard launched at {url}")