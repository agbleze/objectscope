from objectscope.trainer import TrainSession
from objectscope.evaluator import Evaluator
from objectscope import logger
from argparse import ArgumentParser
import os
import pandas as pd

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
    evaluator = Evaluator(cfg=trainer.cfg,
                            test_data_name=args.test_data_name,
                            output_dir=args.output_dir,
                            dataset_nm=trainer.test_data_name,
                            metadata=trainer.test_metadata,
                            roi_heads_score_threshold=args.roi_heads_score_threshold,
                                
                            )
    eval_df = evaluator.evaluate_models(cfg=trainer.cfg)
    eval_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'))
    
if __name__ == "__main__":
    main()
    logger.info("Training and evaluation completed successfully.")    
   
    