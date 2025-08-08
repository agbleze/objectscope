from objectscope.trainer import TrainSession
from objectscope.evaluator import Evaluator
from objectscope import logger
from argparse import ArgumentParser
import os
import pandas as pd
from utils import launch_tensorboard
import subprocess


def parse_args():
    parser = ArgumentParser(description="Setup model training and evaluation parameters")
    parser.add_argument("--train_img_dir", type=str, required=True,
                        help="Directory containing training images"
                        )
    parser.add_argument("--train_coco_json_file", type=str, required=True,
                        help="Path to the COCO JSON file for training"
                        )
    parser.add_argument("--test_img_dir", type=str, required=True,
                        help="Directory containing test images"
                        )
    parser.add_argument("--test_coco_json_file", type=str, required=True,
                        help="Path to the COCO JSON file for testing"
                        )
    parser.add_argument("--config_file_url", type=str, required=True,
                        help="URL of the configuration file"
                        )
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes in the dataset"
                        )
    parser.add_argument("--train_data_name", type=str,
                        help="Name of the training dataset"
                        )
    parser.add_argument("--test_data_name", type=str,
                        help="Name of the testing dataset"
                        )
    parser.add_argument("--train_metadata", type=dict, default={},
                        help="Metadata for the training dataset"
                        )
    parser.add_argument("--test_metadata", type=dict, default={},
                        help="Metadata for the testing dataset"
                        )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output files"
                        )
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of data loading workers"
                        )
    parser.add_argument("--imgs_per_batch", type=int, default=32,
                        help="Number of images per batch"
                        )
    parser.add_argument("--base_lr", type=float, default=0.00005,
                        help="Base learning rate for training"
                        )
    parser.add_argument("--max_iter", type=int, required=True,
                        help="Maximum number of iterations for training"
                        )
    parser.add_argument("--checkpoint_period", type=int, required=True,
                        help="Period for saving checkpoints"
                        )
    parser.add_argument("--start_run", action='store_true',
                        help="Whether to start the training run on initialization"
                        )
    parser.add_argument("--roi_heads_score_threshold", type=float, default=0.5,
                        help="Score threshold for ROI heads during evaluation"
                        )
    parser.add_argument("--tensorboard_logdir", type=str, default="runs",
                        help="Directory for TensorBoard logs"
                        )
    parser.add_argument("--lauch_tensorboard", action="store_true",
                        help="Whether to launch TensorBoard after training. Launches when flag is used"
                        )
    parser.add_argument("--tensorboard_port_num", default="default")
    parser.add_argument("--optimize_model", action="store_true",
                        help="Whether to optimize the model after training"
                        )
    
    return parser.parse_args()

def main():
    args = parse_args()
    trainer = TrainSession(train_img_dir=args.train_img_dir,
                            train_coco_json_file=args.train_coco_json_file,
                            test_img_dir=args.test_img_dir,
                            test_coco_json_file=args.test_coco_json_file,
                            config_file_url=args.config_file_url,
                            num_classes=args.num_classes,
                            train_data_name=args.train_data_name,
                            test_data_name=args.test_data_name,
                            train_metadata=args.train_metadata,
                            test_metadata=args.test_metadata,
                            output_dir=args.output_dir,
                            device=args.device,
                            num_workers=args.num_workers,
                            imgs_per_batch=args.imgs_per_batch,
                            base_lr=args.base_lr,
                            max_iter=args.max_iter,
                            checkpoint_period=args.checkpoint_period,
                            start_run=args.start_run
                        )
    trainer.run()
    if args.lauch_tensorboard:
        launch_tensorboard(logdir=args.tensorboard_logdir, port_num=args.tensorboard_port_num)
        logger.info(f"TensorBoard launched at port {args.tensorboard_port_num}")
    evaluator = Evaluator(cfg=trainer.cfg,
                            test_data_name=trainer.test_data_name,
                            output_dir=args.output_dir,
                            dataset_nm=trainer.test_data_name,
                            metadata=trainer.test_metadata,
                            roi_heads_score_threshold=args.roi_heads_score_threshold,
                                
                            )
    eval_df = evaluator.evaluate_models(cfg=trainer.cfg)
    eval_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'))
    
    if args.optimize_model:
        logger.info("Optimizing model...")
        cmd = ["olive", "auto-opt",
                "--model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct",
                #--trust_remote_code
                "--output_path", args.output_dir,
                "--device", "cpu",
                "--provider", "CPUExecutionProvider",
                "--use_ort_genai",
                "--precision int4",
                "--log_level", 1
                ]
        subprocess.run(cmd, check=True)
        logger.info("Model optimization completed.")
    
if __name__ == "__main__":
    main()
    logger.info("Training and evaluation completed successfully.")    
   
    