import json
import torch
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import huggingface_hub
from huggingface_hub import login
from transformers import TrainingArguments, DataCollatorForLanguageModeling, AutoModel, AutoTokenizer
from trl import SFTTrainer
from peft import LoraConfig
import wandb
import uuid
from datasets import Dataset
from typing import Dict, List, Any
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity






