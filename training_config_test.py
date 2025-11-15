"""
Test Training Configuration for Quick Validation (1-2 Epochs)
This config uses a smaller subset of data for rapid testing
"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """Model-related configuration"""
    base_model_name: str = "EleutherAI/gpt-j-6b"
    model_max_length: int = 512
    use_cache: bool = False
    
@dataclass
class DataConfig:
    """Dataset configuration - REDUCED FOR TESTING"""
    dataset_name: str = "dlnkgpt/dlnkgpt-uncensored-dataset"
    train_split: str = "train"
    validation_split: str = "validation"
    text_column: str = "text"
    max_samples: Optional[int] = 1000  # Use only 1000 samples for quick test
    
@dataclass
class LoRAConfig:
    """LoRA/PEFT configuration"""
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
@dataclass
class TrainingConfig:
    """Training hyperparameters - OPTIMIZED FOR QUICK TEST"""
    output_dir: str = "./training_output_test"
    num_train_epochs: int = 2  # Only 2 epochs for testing
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100  # Reduced warmup for quick test
    
    fp16: bool = True
    bf16: bool = False
    
    logging_dir: str = "./logs_test"
    logging_strategy: str = "steps"
    logging_steps: int = 10
    report_to: str = "tensorboard"
    
    evaluation_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: int = 2  # Keep only 2 checkpoints for testing
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    early_stopping_patience: int = 2  # Reduced for quick test
    early_stopping_threshold: float = 0.001
    
    seed: int = 42
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    
@dataclass
class QuantizationConfig:
    """Quantization settings"""
    use_quantization: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
@dataclass
class QualityAssuranceConfig:
    """Quality assurance - REDUCED TEST SET"""
    run_qa_tests: bool = True
    qa_test_prompts: List[str] = field(default_factory=lambda: [
        "Explain quantum computing in simple terms.",
        "Write a Python function to reverse a string.",
        "What is the meaning of life?",
    ])
    qa_max_new_tokens: int = 150
    qa_temperature: float = 0.7
    qa_top_p: float = 0.9
    qa_do_sample: bool = True
    
@dataclass
class HuggingFaceConfig:
    """Hugging Face Hub configuration - DISABLED FOR TEST"""
    push_to_hub: bool = False  # Don't push test models
    hub_model_id: str = "dlnkgpt/dlnkgpt-uncensored-test"
    hub_strategy: str = "end"
    hub_private_repo: bool = False
    
@dataclass
class WorkflowConfig:
    """Complete workflow configuration for TESTING"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    qa: QualityAssuranceConfig = field(default_factory=QualityAssuranceConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    def __post_init__(self):
        if not self.training.output_dir:
            raise ValueError("training.output_dir must be set")
        
        if self.quantization.load_in_4bit and self.quantization.load_in_8bit:
            print("WARNING: Both 4-bit and 8-bit quantization enabled. Using 4-bit.")
            self.quantization.load_in_8bit = False
    
    def to_dict(self):
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "quantization": self.quantization.__dict__,
            "qa": self.qa.__dict__,
            "huggingface": self.huggingface.__dict__,
        }

# Test configuration instance
test_config = WorkflowConfig()
