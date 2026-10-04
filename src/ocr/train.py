"""
آموزش مدل OCR (CRNN)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import (
    OCR_TRAIN_LABELS, OCR_VAL_LABELS,
    OCR_CHECKPOINT_DIR,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, NUM_CLASSES,
)
from src.ocr.model import CRNN
from src.ocr.dataset import get_dataloaders


def ctc_decode(log_probs):
    """Decode خروجی CTC به متن"""
    from src.config import IDX2CHAR

    # log_probs: (T, batch, num_classes)
    _, max_indices = log_probs.max(2)  # (T, batch)
    max_indices = max_indices.transpose(0, 1)  # (batch, T)

    results = []
    for seq in max_indices:
        text = ""
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:  # 0 = blank
                text += IDX2CHAR.get(idx, "")
            prev = idx
        results.append(text)
    return results


def calculate_cer(preds, targets):
    """محاسبه Character Error Rate"""
    total_distance = 0
    total_chars = 0

    for pred, target in zip(preds, targets):
        # Levenshtein distance
        d = levenshtein(pred, target)
        total_distance += d
        total_chars += len(target)

    return total_distance / max(total_chars, 1)


def levenshtein(s1, s2):
    """فاصله Levenshtein"""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def train():
    # ==================== تنظیمات ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OCR_CHECKPOINT_DIR, exist_ok=True)

    # ==================== DataLoader ====================
    print("\n📥 لود دیتاست...")
    train_loader, val_loader = get_dataloaders(
        OCR_TRAIN_LABELS, OCR_VAL_LABELS, batch_size=BATCH_SIZE
    )
    print(f"   Train: {len(train_loader)} batch")
    print(f"   Val: {len(val_loader)} batch")

    # ==================== مدل ====================
    print("\n🤖 ساخت مدل...")
    model = CRNN(
        input_channel=1,
        num_classes=NUM_CLASSES,
        hidden_size=256,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   پارامترها: {total_params:,}")

    # ==================== Loss & Optimizer ====================
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ==================== TensorBoard ====================
    run_name = f"ocr_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"outputs/logs/{run_name}")
    print(f"\n📊 TensorBoard: outputs/logs/{run_name}")

    # ==================== Training ====================
    best_cer = float('inf')
    best_acc = 0.0

    print(f"\n{'=' * 60}")
    print(f"🚀 شروع آموزش ({EPOCHS} epoch)...")
    print(f"{'=' * 60}\n")

    for epoch in range(1, EPOCHS + 1):
        # ==================== Train ====================
        model.train()
        train_loss = 0.0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels, label_lens in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(imgs)  # (T, batch, num_classes)
            log_probs = F.log_softmax(outputs, dim=2)

            # CTC Loss
            input_lengths = torch.full(
                (imgs.size(0),), log_probs.size(0),
                dtype=torch.long, device=device
            )

            loss = criterion(log_probs, labels, input_lengths, label_lens)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_samples += imgs.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / train_samples

        # ==================== Validation ====================
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []

        from src.config import IDX2CHAR

        with torch.no_grad():
            for imgs, labels, label_lens in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                label_lens = label_lens.to(device)

                outputs = model(imgs)
                log_probs = F.log_softmax(outputs, dim=2)

                input_lengths = torch.full(
                    (imgs.size(0),), log_probs.size(0),
                    dtype=torch.long, device=device
                )

                loss = criterion(log_probs, labels, input_lengths, label_lens)
                val_loss += loss.item() * imgs.size(0)
                val_samples += imgs.size(0)

                # Decode
                preds = ctc_decode(log_probs)

                # Targets
                targets = []
                for lbl, length in zip(labels, label_lens):
                    text = "".join(
                        IDX2CHAR[c.item()] for c in lbl[:length.item()]
                    )
                    targets.append(text)

                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_val_loss = val_loss / val_samples

        # ==================== Metrics ====================
        correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
        seq_acc = correct / len(all_targets)
        cer = calculate_cer(all_preds, all_targets)

        # ==================== Log ====================
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Seq Accuracy: {seq_acc * 100:.2f}%")
        print(f"  CER: {cer * 100:.2f}%")
        print(f"  نمونه: Pred='{all_preds[0]}' | GT='{all_targets[0]}'\n")

        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/SeqAcc", seq_acc, epoch)
        writer.add_scalar("Val/CER", cer, epoch)

        scheduler.step(avg_val_loss)

        # ==================== Checkpoint ====================
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_cer": best_cer,
            "best_acc": best_acc,
            "num_classes": NUM_CLASSES,
        }

        # بهترین مدل بر اساس CER
        if cer < best_cer:
            best_cer = cer
            best_acc = seq_acc
            checkpoint["best_cer"] = best_cer
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, os.path.join(OCR_CHECKPOINT_DIR, "best.pt"))
            print(f"  ✅ بهترین مدل ذخیره شد (CER={cer * 100:.2f}%, Acc={seq_acc * 100:.2f}%)")

        # آخرین مدل
        torch.save(checkpoint, os.path.join(OCR_CHECKPOINT_DIR, "last.pt"))

    writer.close()

    print(f"\n{'=' * 60}")
    print(f"✅ آموزش تمام شد!")
    print(f"   بهترین Seq Acc: {best_acc * 100:.2f}%")
    print(f"   بهترین CER: {best_cer * 100:.2f}%")
    print(f"   مدل: {OCR_CHECKPOINT_DIR}/best.pt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()