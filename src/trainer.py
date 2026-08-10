import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from .data_loader import collate_with_features

class EnhancedPromptLapFormerTrainer:
    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        self.lambda_lap = 0.001
        self.lambda_prompt = 0.001

    def train_epoch(self, dataset, epoch):
        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_with_features,
            num_workers=0
        )

        total_loss = 0
        y_true, y_pred = [], []
        num_batches = 0

        for batch_data in loader:
            batch_data = batch_data.to(next(self.model.parameters()).device)

            h_p_precomputed = None
            if hasattr(batch_data, 'embeddings') and batch_data.embeddings is not None:
                h_p_precomputed = batch_data.embeddings

            self.optimizer.zero_grad()

            outputs = self.model(
                batch_data,
                return_all_losses=True,
                h_p_precomputed=h_p_precomputed
            )

            logits = outputs['logits']
            cls_loss = F.cross_entropy(logits, batch_data.y)

            lap_loss = torch.abs(outputs['lap_loss'])
            prompt_loss = torch.abs(outputs['prompt_loss'])

            loss = cls_loss + self.lambda_lap * lap_loss + self.lambda_prompt * prompt_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            y_true.extend(batch_data.y.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
            num_batches += 1

        self.scheduler.step()

        accuracy = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        return total_loss / max(num_batches, 1), accuracy

    def train(self, train_dataset, val_dataset, test_dataset, epochs=150):
        tester = EnhancedPromptLapFormerTester(self.model)

        print("=" * 120)
        print("Training Enhanced PromptLapFormer for BACE Classification")
        print("=" * 120)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print("=" * 120)

        header = f"{'Epoch':>6} {'Time(sec)':>10} {'Train_Loss':>12} {'Train_Acc':>12} {'Val_Acc':>12} {'Val_AUC':>12}"
        print(header)
        print("=" * 120)

        best_val_auc = 0
        best_val_acc = 0

        epoch_results = []

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            try:
                train_loss, train_acc = self.train_epoch(train_dataset, epoch)

                val_metrics = tester.test(val_dataset)

                epoch_time = time.time() - epoch_start

                print(f"{epoch:6d} {epoch_time:10.2f} {train_loss:12.6f} {train_acc:12.6f} {val_metrics['accuracy']:12.6f} {val_metrics['auc_roc']:12.6f}")

                epoch_results.append({
                    'epoch': epoch,
                    'time_sec': epoch_time,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_auc': val_metrics['auc_roc'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1']
                })

                if val_metrics['auc_roc'] > 0.5 and val_metrics['auc_roc'] > best_val_auc:
                    best_val_auc = val_metrics['auc_roc']
                    torch.save(self.model.state_dict(), 'best_model_enhanced_bace.pt')
                elif val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    torch.save(self.model.state_dict(), 'best_model_enhanced_bace.pt')

            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                continue

        total_time = time.time() - start_time

        print("=" * 120)
        print(f"Training finished! Total time: {total_time:.2f} seconds")
        print("=" * 120)

        if len(epoch_results) > 0:
            df_results = pd.DataFrame(epoch_results)
            df_results.to_csv('Enhanced_PromptLapFormer_BACE_Training_Metrics.csv', index=False)
            print("Training metrics saved to 'Enhanced_PromptLapFormer_BACE_Training_Metrics.csv'")

        if os.path.exists('best_model_enhanced_bace.pt'):
            self.model.load_state_dict(torch.load('best_model_enhanced_bace.pt'))
            print("Loaded best model")

        print("\n" + "=" * 120)
        print("Evaluating on test set...")
        print("=" * 120)
        test_metrics = tester.test(test_dataset)

        print(f"Test Loss: {test_metrics['loss']:.6f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.6f}")
        print(f"Test AUC-ROC: {test_metrics['auc_roc']:.6f}")
        print(f"Test Precision: {test_metrics['precision']:.6f}")
        print(f"Test Recall: {test_metrics['recall']:.6f}")
        print(f"Test F1: {test_metrics['f1']:.6f}")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_metrics['y_true'], test_metrics['y_pred'])
        print(cm)

        test_df = pd.DataFrame([{
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc_roc'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1']
        }])
        test_df.to_csv('Enhanced_PromptLapFormer_BACE_Test_Metrics.csv', index=False)
        print("Test metrics saved to 'Enhanced_PromptLapFormer_BACE_Test_Metrics.csv'")

        return {
            'epoch_results': epoch_results,
            'test_metrics': test_metrics
        }

class EnhancedPromptLapFormerTester:
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_with_features,
            num_workers=0
        )

        total_loss = 0
        y_true, y_pred, y_probs = [], [], []
        num_batches = 0

        with torch.no_grad():
            for batch_data in loader:
                batch_data = batch_data.to(next(self.model.parameters()).device)

                h_p_precomputed = None
                if hasattr(batch_data, 'embeddings') and batch_data.embeddings is not None:
                    h_p_precomputed = batch_data.embeddings

                outputs = self.model(
                    batch_data,
                    return_all_losses=True,
                    h_p_precomputed=h_p_precomputed
                )
                logits = outputs['logits']

                cls_loss = F.cross_entropy(logits, batch_data.y)
                total_loss += cls_loss.item()

                y_true.extend(batch_data.y.cpu().numpy())
                y_pred.extend(logits.argmax(dim=1).cpu().numpy())
                y_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
                num_batches += 1

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)

        accuracy = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0

        try:
            if len(np.unique(y_true)) > 1:
                auc_roc = roc_auc_score(y_true, y_probs)
            else:
                auc_roc = 0.5
        except:
            auc_roc = 0.5

        if len(np.unique(y_pred)) > 1:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0

            if len(np.unique(y_true)) == 1:
                if y_pred[0] == y_true[0]:
                    precision = 1.0
                    recall = 1.0
                    f1 = 1.0

        return {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs
        }
