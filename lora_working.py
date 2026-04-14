cat > ~ / dialect - project / scripts / lora_train_v5.py << 'EOF'
import torch
import torch.nn as nn
import os
import sys
import wave
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict
from peft import LoraConfig, get_peft_model
from qwen_asr import Qwen3ASRModel

# === CONFIG ===
SAMPLES_PER_DIALECT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
EPOCHS = 10
LR = 3e-4  # Increased learning rate
AUDIO_DIR = "/mnt/c/Users/karam/PyCharmMiscProject/adi17_audio"
SAVE_DIR = "/mnt/c/Users/karam/PyCharmMiscProject/results"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"=== LoRA Training: {SAMPLES_PER_DIALECT}/dialect ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Saving to: {SAVE_DIR}")

# Load model
print("Loading model...")
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.float32,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)
processor = model.processor
device = "cuda:0"
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# Dataset class
class DialectAudioDataset(Dataset):
    def __init__(self, file_list, labels, audio_dir, processor):
        self.file_list = file_list
        self.labels = labels
        self.audio_dir = audio_dir
        self.processor = processor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.audio_dir, self.file_list[idx])
        with wave.open(filepath, 'r') as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            sr = w.getframerate()
        feats = self.processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        return feats['input_features'].squeeze(0), self.labels[idx]


# LoRA model
class DialectLoRAModel(nn.Module):
    def __init__(self, encoder, hidden_dim=1024, num_classes=17):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_features, feature_lens):
        # Pass the 2D tensor to avoid Qwen3 .T broadcasting issues
        audio_output = self.encoder(
            input_features[0, :, :feature_lens[0]],
            feature_lens=feature_lens,
        )

        # Pool across the Sequence Dimension (dim=0)
        pooled = audio_output.last_hidden_state.mean(dim=0)

        # Add the batch dimension back for the Linear classification head
        return self.head(pooled.unsqueeze(0))


# Prepare data
files = os.listdir(AUDIO_DIR)
dialect_to_idx = {}
for f in files:
    d = f.split("_", 1)[0]
    if d not in dialect_to_idx:
        dialect_to_idx[d] = len(dialect_to_idx)

idx_to_dialect = {v: k for k, v in dialect_to_idx.items()}
file_labels = [dialect_to_idx[f.split("_", 1)[0]] for f in files]

# Split all files
train_files, temp_files, train_labels, temp_labels = train_test_split(
    files, file_labels, test_size=0.2, random_state=42, stratify=file_labels)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

# Subsample training data
dialect_files = defaultdict(list)
for f, l in zip(train_files, train_labels):
    dialect_files[l].append(f)

sub_files, sub_labels = [], []
for d_idx, flist in dialect_files.items():
    take = min(len(flist), SAMPLES_PER_DIALECT)
    sub_files.extend(flist[:take])
    sub_labels.extend([d_idx] * take)

sub_train_ds = DialectAudioDataset(sub_files, sub_labels, AUDIO_DIR, processor)
val_ds = DialectAudioDataset(val_files, val_labels, AUDIO_DIR, processor)
test_ds = DialectAudioDataset(test_files, test_labels, AUDIO_DIR, processor)

print(f"Train: {len(sub_train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
print(f"Dialects: {dialect_to_idx}")

# Build LoRA model
encoder = model.model.thinker.audio_tower
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
lora_m = DialectLoRAModel(encoder, hidden_dim=1024, num_classes=17).to(device)
lora_m.encoder = get_peft_model(lora_m.encoder, lora_config)

trainable = sum(p.numel() for p in lora_m.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in lora_m.parameters())
print(f"Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.2f}%)")

# Train
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_m.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss()
best_val_acc = 0

for epoch in range(EPOCHS):
    # Lock base encoder in eval mode to prevent representation shift
    lora_m.eval()

    # Enable training ONLY for the new head and LoRA weights
    lora_m.head.train()
    for name, module in lora_m.named_modules():
        if 'lora' in name.lower():
            module.train()

    total_loss, correct, total = 0, 0, 0

    # FIX: Shuffle indices to prevent catastrophic forgetting of classes
    indices = list(range(len(sub_train_ds)))
    random.shuffle(indices)

    for step, i in enumerate(indices):
        input_feats, label = sub_train_ds[i]
        input_feats = input_feats.unsqueeze(0).to(device).to(torch.float32)

        # Force gradients to flow back to the LoRA layers
        input_feats.requires_grad_(True)

        feature_len = torch.tensor([input_feats.shape[-1]], device=device)
        label_t = torch.tensor([label], device=device)

        optimizer.zero_grad()
        logits = lora_m(input_feats, feature_len)
        loss = criterion(logits, label_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_m.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == label_t).sum().item()
        total += 1

        if (step + 1) % 500 == 0:
            print(
                f"  Epoch {epoch + 1}, sample {step + 1}/{len(sub_train_ds)}, loss={total_loss / total:.4f}, acc={correct / total:.4f}")

    # Validate
    lora_m.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for i in range(len(val_ds)):
            input_feats, label = val_ds[i]
            input_feats = input_feats.unsqueeze(0).to(device).to(torch.float32)
            feature_len = torch.tensor([input_feats.shape[-1]], device=device)
            logits = lora_m(input_feats, feature_len)
            val_correct += (logits.argmax(1).item() == label)
            val_total += 1

    val_acc = val_correct / val_total
    print(
        f"Epoch {epoch + 1}: loss={total_loss / len(sub_train_ds):.4f}, train_acc={correct / total:.4f}, val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(lora_m.state_dict(), f"{SAVE_DIR}/best_lora_{SAMPLES_PER_DIALECT}pd.pt")
        print(f"  New best! Saved.")

print(f"\nBest val accuracy: {best_val_acc:.4f}")

# Test
lora_m.load_state_dict(torch.load(f"{SAVE_DIR}/best_lora_{SAMPLES_PER_DIALECT}pd.pt"))
lora_m.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for i in range(len(test_ds)):
        input_feats, label = test_ds[i]
        input_feats = input_feats.unsqueeze(0).to(device).to(torch.float32)
        feature_len = torch.tensor([input_feats.shape[-1]], device=device)
        logits = lora_m(input_feats, feature_len)
        all_preds.append(logits.argmax(1).item())
        all_true.append(label)

labels = [idx_to_dialect[i] for i in range(17)]
test_acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
print(f"\nLoRA Test Accuracy ({SAMPLES_PER_DIALECT}/dialect): {test_acc:.4f}")
print(classification_report(all_true, all_preds, target_names=labels, zero_division=0))
EOF

python3
~ / dialect - project / scripts / lora_train_v5.py