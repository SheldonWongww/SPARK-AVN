import os
import sys
import warnings
import logging
import glob
import inspect

# Suppress various warnings - MUST be done before importing other modules
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress specific warnings by message content
warnings.filterwarnings("ignore", message=".*Unable to register.*")
warnings.filterwarnings("ignore", message=".*PulseAudio.*")
warnings.filterwarnings("ignore", message=".*PortAudio.*")
warnings.filterwarnings("ignore", message=".*computation placer already registered.*")

# Suppress TensorFlow/XLA warnings at C++ level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress ALSA audio warnings
os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'
os.environ['ALSA_PCM_SUBDIR'] = '0'

# Suppress PortAudio warnings
os.environ['PORTAUDIO_DEVICE'] = '0'

# Suppress specific TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)


class WarningSuppressor:
    def __init__(self):
        self.original_stderr = sys.stderr
        
    def write(self, message):
        # Filter out specific warning messages
        if any(pattern in message for pattern in [
            'computation placer already registered',
            'PulseAudio: Unable to connect',
            'Unable to register',
            'cuFFT factory',
            'cuDNN factory',
            'cuBLAS factory'
        ]):
            return  # Suppress these messages
        self.original_stderr.write(message)
        
    def flush(self):
        self.original_stderr.flush()
        
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Only suppress stderr if we're not in debug mode
if not any('--debug' in arg for arg in sys.argv):
    sys.stderr = WarningSuppressor()

import argparse
import torch
import numpy as np
import random

# Import baseline registry first
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.tensorboard_utils import TensorboardWriter

# Import configuration
from cfgs.default import get_config

# Import trainers to ensure registration
from cavn.engin.base_ddppo_trainer import BaseDDPPOTrainer
from cavn.engin.continual_ddppo_trainer import ContinualDDPPOTrainer

# Import environment and task to ensure registration
from cavn.env.simulator import SoundEventNavSim
from cavn.env.sound_eventnav_dataset import SoundEventNavDataset
from cavn.env.sound_eventnav_task import SoundEventNavigationTask


def find_best_ckpt_idx(tb_dir, metric='spl', min_step=-1, max_step=10000):
    """
    Find the best checkpoint index from TensorBoard logs
    """
    try:
        import tensorflow as tf
        tf_version = int(tf.__version__.split('.')[0])
        
        max_value = 0
        max_index = -1
        
        events = os.listdir(tb_dir)
        
        for event in events:
            if "events" not in event:
                continue
                
            event_path = os.path.join(tb_dir, event)
            
            # Handle different TensorFlow versions
            if tf_version >= 2:
                iterator = tf.compat.v1.train.summary_iterator(event_path)
            else:
                iterator = tf.train.summary_iterator(event_path)
            
            for e in iterator:
                if len(e.summary.value) == 0:
                    continue
                    
                # Look for validation metrics
                if not e.summary.value[0].tag.startswith('val'):
                    continue
                    
                # Check if it's the metric we're looking for
                if metric not in e.summary.value[0].tag:
                    continue
                    
                # Skip soft-spl if looking for spl
                if metric == 'spl' and 'softspl' in e.summary.value[0].tag:
                    continue
                    
                if not min_step <= e.step <= max_step:
                    continue
                    
                if e.summary.value[0].simple_value > max_value:
                    max_value = e.summary.value[0].simple_value
                    max_index = e.step
        
        if max_index == -1:
            logging.warning(f'No best checkpoint found in {tb_dir}')
        else:
            logging.info(f'Best checkpoint at step {max_index} with {metric}={max_value:.4f}')
            
        return max_index
        
    except ImportError:
        logging.warning("TensorFlow not installed, cannot find best checkpoint automatically")
        return -1


def main():
    parser = argparse.ArgumentParser(description="Continual Learning Audio-Visual Navigation")
    
    # Main arguments
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "eval-pretrained", "eval-continual"],
        default="train",
        help="Run type: train, eval, eval-pretrained (evaluate pretrained model on all domains), or eval-continual (evaluate continual learning checkpoint on seen/next domains)"
    )
    
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="For eval-pretrained: evaluate domains in reverse order (19->0) instead of forward order (0->19)"
    )
    
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="Path to experiment config yaml (e.g., cfgs/cl_exp/single_source/finetune/train.yaml)"
    )
    
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory to store models and logs"
    )
    
    # Training arguments
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Overwrite existing model directory"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval of checkpoints"
    )
    
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Previous checkpoint index for evaluation"
    )
    
    parser.add_argument(
        "--max-ckpt-ind",
        type=int,
        default=10000,
        help="Maximum checkpoint index for evaluation"
    )
    
    parser.add_argument(
        "--eval-best",
        default=False,
        action='store_true',
        help="Evaluate the best checkpoint based on validation SPL"
    )
    
    parser.add_argument(
        "--use-last-ckpt",
        default=False,
        action='store_true',
        help="Use the last checkpoint for evaluation"
    )

    parser.add_argument(
        "--continual-eval-mode",
        type=str,
        default="final",
        choices=["final", "matrix"],
        help="For eval-continual: final (only final checkpoint) or matrix (full performance matrix)"
    )
    
    # Model configuration
    parser.add_argument(
        "--decoder-type",
        type=str,
        default=None,
        help="Decoder type (e.g., MSMT, TRANSFORMER)"
    )
    
    parser.add_argument(
        "--norm-first",
        type=bool,
        default=None,
        help="Whether to normalize first in transformer layers"
    )
    
    parser.add_argument(
        "--actor-critic-path",
        type=str,
        default="-1",
        help="Path to pretrained actor-critic model"
    )
    
    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced episodes"
    )
    
    # Additional config options
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s, %(levelname)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Load configuration
    config = get_config(
        config_paths=args.exp_config,
        opts=args.opts,
        model_dir=args.model_dir,
        run_type=args.run_type,
        overwrite=args.overwrite,
        decoder_type=args.decoder_type,
        norm_first=args.norm_first,
        actor_critic_path=args.actor_critic_path,
    )
    
    # Override configuration with command line arguments
    if args.seed is not None:
        config.defrost()
        config.SEED = args.seed
        config.TASK_CONFIG.SEED = args.seed
        config.freeze()
    
    # Debug mode adjustments
    if args.debug:
        config.defrost()
        config.NUM_UPDATES = 10
        config.NUM_UPDATES_PER_DOMAIN = 10
        config.TEST_EPISODE_COUNT = 10
        config.LOG_INTERVAL = 1
        config.CHECKPOINT_INTERVAL = 5
        config.DEBUG = True
        config.freeze()
        logging.info("Debug mode enabled - reduced episodes and updates")
    
    # Create model directory if it doesn't exist
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Log configuration info
    logging.info(f"Running {args.run_type} with config: {args.exp_config}")
    logging.info(f"Model directory: {config.MODEL_DIR}")
    logging.info(f"Trainer: {config.TRAINER_NAME}")
    if hasattr(config, 'CL_METHOD'):
        logging.info(f"CL Method: {config.CL_METHOD.TYPE}")
    logging.info(f"Seed: {config.SEED}")
    
    # Initialize trainer
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"Trainer {config.TRAINER_NAME} not found in registry"
    
    trainer = trainer_init(config)
    
    # Set PyTorch thread count
    torch.set_num_threads(1)
    
    # Execute based on run type
    if args.run_type == "train":
        logging.info("Starting training...")
        trainer.train()
    
    elif args.run_type == "eval-pretrained":
        # Evaluate pretrained model on all 20 domains to compute R_{0,i}
        logging.info("Evaluating pretrained model on all domains...")
        if args.reverse_order:
            logging.info("Using REVERSE order (19->0)")
        else:
            logging.info("Using FORWARD order (0->19)")
        
        # Check if trainer has the method
        if hasattr(trainer, 'eval_pretrained_on_all_domains'):
            trainer.eval_pretrained_on_all_domains(reverse_order=args.reverse_order)
        else:
            logging.error(f"Trainer {config.TRAINER_NAME} does not support eval-pretrained mode")
            logging.error(f"This mode is only available for continual_ddppo trainer")
    
    elif args.run_type == "eval-continual":
        # Evaluate continual learning checkpoint on seen domains and next domain
        logging.info("Evaluating continual learning checkpoint...")
        logging.info(f"Continual evaluation mode: {args.continual_eval_mode}")
        
        # Check if trainer has the method
        if hasattr(trainer, 'evaluate_continual'):
            eval_fn = trainer.evaluate_continual
            eval_sig = inspect.signature(eval_fn)
            if "eval_mode" in eval_sig.parameters:
                eval_fn(eval_mode=args.continual_eval_mode)
            else:
                if args.continual_eval_mode != "final":
                    logging.warning(
                        f"Trainer {config.TRAINER_NAME} does not support eval_mode argument, "
                        "fallback to default evaluate_continual() behavior."
                    )
                eval_fn()
        else:
            logging.error(f"Trainer {config.TRAINER_NAME} does not support eval-continual mode")
            logging.error(f"This mode is only available for continual_ddppo trainer")
            
    elif args.run_type == "eval":
        # Handle evaluation
        if args.eval_best:
            # Find best checkpoint from TensorBoard logs
            tb_dir = os.path.join(config.MODEL_DIR, 'tb')
            best_ckpt_idx = find_best_ckpt_idx(
                tb_dir, 
                max_step=args.max_ckpt_ind
            )
            if best_ckpt_idx >= 0:
                eval_ckpt_path = os.path.join(
                    config.CHECKPOINT_FOLDER,
                    f'ckpt.{best_ckpt_idx}.pth'
                )
                logging.info(f'Evaluating best checkpoint: {eval_ckpt_path}')
                config.defrost()
                config.EVAL_CKPT_PATH_DIR = eval_ckpt_path
                config.freeze()
        
        # Run evaluation
        if hasattr(trainer, 'eval'):
            # Use ENMuS-style eval method if available
            trainer.eval(
                eval_interval=args.eval_interval,
                prev_ckpt_ind=args.prev_ckpt_ind,
                use_last_ckpt=args.use_last_ckpt
            )
        else:
            # Find checkpoints to evaluate
            if hasattr(config, 'EVAL_CKPT_PATH_DIR') and os.path.exists(config.EVAL_CKPT_PATH_DIR):
                checkpoint_paths = [config.EVAL_CKPT_PATH_DIR]
            else:
                checkpoint_paths = sorted(
                    glob.glob(os.path.join(config.CHECKPOINT_FOLDER, "*.pth")),
                    key=lambda x: int(x.split('.')[-2].split('_')[-1]) if '_' in x else int(x.split('.')[-2])
                )
                
                if args.use_last_ckpt and checkpoint_paths:
                    checkpoint_paths = [checkpoint_paths[-1]]
            
            if not checkpoint_paths:
                logging.error(f"No checkpoints found in {config.CHECKPOINT_FOLDER}")
                return
            
            # Evaluate each checkpoint
            with TensorboardWriter(config.TENSORBOARD_DIR, flush_secs=30) as writer:
                for ckpt_idx, ckpt_path in enumerate(checkpoint_paths):
                    logging.info(f"Evaluating checkpoint: {ckpt_path}")
                    
                    # Extract checkpoint index from filename for TensorBoard x-axis
                    # Format: ckpt.X.pth or ckpt_domain_X.pth
                    try:
                        ckpt_filename = os.path.basename(ckpt_path)
                        if '_' in ckpt_filename:
                            # Format: ckpt_domain_X.pth or similar
                            checkpoint_index = int(ckpt_filename.split('.')[-2].split('_')[-1])
                        else:
                            # Format: ckpt.X.pth
                            checkpoint_index = int(ckpt_filename.split('.')[-2])
                    except:
                        # Fallback to enumerate index if parsing fails
                        checkpoint_index = ckpt_idx
                        logging.warning(f"Could not parse checkpoint index from filename, using {checkpoint_index}")
                    
                    # Evaluate on validation set
                    metrics = trainer._eval_checkpoint(
                        checkpoint_path=ckpt_path,
                        writer=writer,
                        checkpoint_index=checkpoint_index
                    )
                    
                    # Log results
                    logging.info(f"Evaluation results for {ckpt_path} (checkpoint {checkpoint_index}):")
                    for metric_name, metric_value in metrics.items():
                        logging.info(f"  {metric_name}: {metric_value:.4f}")
    
    logging.info("Done!")


if __name__ == "__main__":
    main()
