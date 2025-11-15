"""
Training Configuration for dLNk GPT Enhanced Workflow
All hyperparameters and settings in one place for easy modification
"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """Model-related configuration"""
    base_model_name: str = "EleutherAI/gpt-j-6b"
    model_max_length: int = 512
    use_cache: bool = False  # Disable for training
    
@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "dlnkgpt/dlnkgpt-uncensored-dataset"
    train_split: str = "train"
    validation_split: str = "validation"
    text_column: str = "text"
    max_samples: Optional[int] = None  # None = use all data
    
@dataclass
class LoRAConfig:
    """LoRA/PEFT configuration"""
    use_lora: bool = True
    lora_r: int = 16  # Rank of LoRA matrices
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05  # Dropout for regularization
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Basic settings
    output_dir: str = "./training_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch size = 32
    
    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 500
    
    # Precision and optimization
    fp16: bool = True  # Use mixed precision on GPU
    bf16: bool = False  # Use bfloat16 if available (A100)
    
    # Logging
    logging_dir: str = "./logs"
    logging_strategy: str = "steps"
    logging_steps: int = 10
    report_to: str = "tensorboard"
    
    # Evaluation
    eval_strategy: str = "epoch"
    eval_steps: Optional[int] = None  # None = evaluate at end of epoch
    
    # Checkpointing
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: int = 3  # Keep only last 3 checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False  # Lower eval_loss is better
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Other
    seed: int = 42
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    
@dataclass
class QuantizationConfig:
    """Quantization settings for memory optimization"""
    use_quantization: bool = True
    load_in_8bit: bool = False  # Use 8-bit quantization
    load_in_4bit: bool = True  # Use 4-bit quantization (more aggressive)
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
@dataclass
class QualityAssuranceConfig:
    """Quality assurance and testing configuration"""
    run_qa_tests: bool = True
    qa_test_prompts: List[str] = field(default_factory=lambda: [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to calculate the factorial of a number.",
        "What are the main causes of climate change?",
        "Write a hello world program in Python.",
        "Explain what is machine learning.",
    ])
    qa_max_new_tokens: int = 200
    qa_temperature: float = 0.7
    qa_top_p: float = 0.9
    qa_do_sample: bool = True
    
@dataclass
class HuggingFaceConfig:
    """Hugging Face Hub configuration"""
    push_to_hub: bool = True
    hub_model_id: str = "dlnkgpt/dlnkgpt-uncensored"
    hub_strategy: str = "end"  # Push at the end of training
    hub_private_repo: bool = False
    
@dataclass
class WorkflowConfig:
    """Complete workflow configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    qa: QualityAssuranceConfig = field(default_factory=QualityAssuranceConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Ensure output directories are set
        if not self.training.output_dir:
            raise ValueError("training.output_dir must be set")
        
        # Warn if using both 4-bit and 8-bit
        if self.quantization.load_in_4bit and self.quantization.load_in_8bit:
            print("WARNING: Both 4-bit and 8-bit quantization enabled. Using 4-bit.")
            self.quantization.load_in_8bit = False
    
    def to_dict(self):
        """Convert configuration to dictionary for logging"""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "quantization": self.quantization.__dict__,
            "qa": self.qa.__dict__,
            "huggingface": self.huggingface.__dict__,
        }

# Default configuration instance
default_config = WorkflowConfig()
