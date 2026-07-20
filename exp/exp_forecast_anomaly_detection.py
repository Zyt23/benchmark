from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import csv
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Forecast_Anomaly_Detection(Exp_Basic):
    """Use a forecasting head for anomaly detection.

    The model is built as a normal long-term forecasting model.  It is trained
    only on normal samples.  At evaluation time, each window receives one
    anomaly score: the prediction error on the forecast horizon.  A threshold is
    selected without touching the test set, then the held-out test split is
    evaluated as binary anomaly detection.
    """

    def __init__(self, args):
        self.external_task_name = args.task_name
        args.task_name = 'long_term_forecast'
        super(Exp_Forecast_Anomaly_Detection, self).__init__(args)
        args.task_name = self.external_task_name

    def _build_model(self):
        original = self.args.task_name
        self.args.task_name = 'long_term_forecast'
        model = self.model_dict[self.args.model](self.args).float()
        self.args.task_name = original

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _forward_forecast(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        true = batch_y[:, -self.args.pred_len:, f_dim:]
        return outputs, true

    def vali(self, vali_data, vali_loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                outputs, true = self._forward_forecast(batch_x, batch_y, batch_x_mark, batch_y_mark)
                losses.append(criterion(outputs, true).item())
        self.model.train()
        return float(np.average(losses)) if losses else float('inf')

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                outputs, true = self._forward_forecast(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(outputs, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = float(np.average(train_loss)) if train_loss else float('inf')
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def _collect_scores(self, loader):
        score_parts = {
            'mse': [],
            'mae': [],
            'channel_mse_max': [],
            'time_mse_max': [],
        }
        labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_label = batch
                outputs, true = self._forward_forecast(batch_x, batch_y, batch_x_mark, batch_y_mark)
                err = outputs - true
                squared = err ** 2
                score_parts['mse'].append(
                    torch.mean(squared, dim=(1, 2)).detach().cpu().numpy())
                score_parts['mae'].append(
                    torch.mean(torch.abs(err), dim=(1, 2)).detach().cpu().numpy())
                score_parts['channel_mse_max'].append(
                    torch.mean(squared, dim=1).amax(dim=1).detach().cpu().numpy())
                score_parts['time_mse_max'].append(
                    torch.mean(squared, dim=2).amax(dim=1).detach().cpu().numpy())
                labels.append(batch_label.detach().cpu().numpy().reshape(-1))
        scores = {
            name: np.concatenate(parts, axis=0).astype(np.float64, copy=False)
            for name, parts in score_parts.items()
        }
        return scores, np.concatenate(labels, axis=0).astype(int)

    @staticmethod
    def _best_f1_threshold(scores, labels, score_name):
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores).astype(float)
        if labels.max(initial=0) == 0 or scores.size == 0:
            return None, None, []

        percentiles = np.concatenate([
            np.linspace(50.0, 95.0, 46),
            np.linspace(95.5, 99.5, 9),
            np.asarray([99.7, 99.9]),
        ])
        candidates = np.unique(np.percentile(scores, percentiles))
        best = None
        sweep_rows = []
        for threshold in candidates:
            pred = (scores > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, pred, average='binary', zero_division=0)
            acc = accuracy_score(labels, pred)
            balanced_acc = balanced_accuracy_score(labels, pred)
            current = (
                float(f1), float(balanced_acc), float(recall),
                float(precision), float(acc), float(threshold),
            )
            sweep_rows.append({
                'score': score_name,
                'threshold': float(threshold),
                'val_accuracy': float(acc),
                'val_balanced_accuracy': float(balanced_acc),
                'val_precision': float(precision),
                'val_recall': float(recall),
                'val_f1': float(f1),
            })
            if best is None or current[:4] > best[:4]:
                best = current
        if best is None:
            return None, None, sweep_rows
        f1, balanced_acc, recall, precision, acc, threshold = best
        return threshold, {
            'val_accuracy': acc,
            'val_balanced_accuracy': balanced_acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
        }, sweep_rows

    def _select_threshold(self, train_scores, val_scores, threshold_scores,
                          threshold_labels, score_name):
        source = getattr(self.args, 'anomaly_threshold_source', 'val')
        percentile = float(getattr(self.args, 'anomaly_threshold_percentile', 99.0))

        if source == 'val_mixed_best_f1':
            threshold, val_metrics, sweep_rows = self._best_f1_threshold(
                threshold_scores, threshold_labels, score_name)
            if threshold is not None:
                return float(threshold), source, percentile, val_metrics, sweep_rows
            source = 'val'

        if source == 'train':
            return float(np.percentile(train_scores, percentile)), source, percentile, {}, []
        if source == 'combined':
            combined = np.concatenate([train_scores, val_scores], axis=0)
            return float(np.percentile(combined, percentile)), source, percentile, {}, []
        return float(np.percentile(val_scores, percentile)), 'val', percentile, {}, []

    def _select_score_and_threshold(self, train_scores, val_scores,
                                    threshold_scores, threshold_labels):
        requested = str(getattr(self.args, 'forecast_anomaly_score', 'auto')).lower()
        source = str(getattr(self.args, 'anomaly_threshold_source', 'val'))
        if requested == 'auto' and source == 'val_mixed_best_f1':
            candidate_names = list(train_scores)
        elif requested == 'auto':
            candidate_names = ['mse']
        else:
            candidate_names = [requested]

        choices = []
        all_sweep_rows = []
        for score_name in candidate_names:
            threshold, threshold_source, percentile, val_metrics, sweep_rows = self._select_threshold(
                train_scores[score_name],
                val_scores[score_name],
                threshold_scores[score_name],
                threshold_labels,
                score_name,
            )
            all_sweep_rows.extend(sweep_rows)
            rank_key = (
                float(val_metrics.get('val_f1', -1.0)),
                float(val_metrics.get('val_balanced_accuracy', -1.0)),
                float(val_metrics.get('val_recall', -1.0)),
                float(val_metrics.get('val_precision', -1.0)),
            )
            choices.append((rank_key, score_name, threshold, threshold_source,
                            percentile, val_metrics))

        _, score_name, threshold, threshold_source, percentile, val_metrics = max(
            choices, key=lambda item: item[0])
        for row in all_sweep_rows:
            row['selected'] = int(
                row['score'] == score_name
                and np.isclose(float(row['threshold']), float(threshold)))
        return (score_name, float(threshold), threshold_source, percentile,
                val_metrics, all_sweep_rows)

    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        threshold_data, threshold_loader = self._get_data(flag='threshold')
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'),
                map_location=self.device,
            ))

        train_score_map, _ = self._collect_scores(train_loader)
        val_score_map, _ = self._collect_scores(val_loader)
        threshold_score_map, thre_labels = self._collect_scores(threshold_loader)
        test_score_map, gt = self._collect_scores(test_loader)

        (score_name, threshold, threshold_source, threshold_percentile,
         val_metrics, sweep_rows) = self._select_score_and_threshold(
            train_score_map, val_score_map, threshold_score_map, thre_labels)
        train_scores = train_score_map[score_name]
        val_scores = val_score_map[score_name]
        test_scores = test_score_map[score_name]

        pred = (test_scores > threshold).astype(int)
        gt = np.asarray(gt).astype(int)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0)
        balanced_accuracy = balanced_accuracy_score(gt, pred)
        macro_f1 = f1_score(gt, pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(gt, pred, average='weighted', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
        if np.unique(gt).size == 2:
            auroc = float(roc_auc_score(gt, test_scores))
            auprc = float(average_precision_score(gt, test_scores))
        else:
            auroc = float('nan')
            auprc = float('nan')
        true_counts = np.bincount(gt.astype(int), minlength=2)
        pred_counts = np.bincount(pred.astype(int), minlength=2)

        print("Threshold: {}".format(threshold))
        print("Threshold source: {}, percentile: {}, score: {}".format(
            threshold_source, threshold_percentile, score_name))
        if val_metrics:
            print("Threshold validation metrics: {}".format(val_metrics))
        print("Accuracy: {:0.4f}, balanced accuracy: {:0.4f}, macro F1: {:0.4f}".format(
            accuracy, balanced_accuracy, macro_f1))
        print("Precision: {:0.4f}, Recall: {:0.4f}, anomaly F1: {:0.4f}, AUROC: {:0.4f}, AUPRC: {:0.4f}".format(
            precision, recall, f_score, auroc, auprc))
        print("true_counts: {}, pred_counts: {}, TN/FP/FN/TP: {}/{}/{}/{}".format(
            true_counts.tolist(), pred_counts.tolist(), int(tn), int(fp), int(fn), int(tp)))

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        metric_row = {
            'setting': setting,
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f_score),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'specificity': specificity,
            'auroc': auroc,
            'auprc': auprc,
            'true_counts': true_counts.tolist(),
            'pred_counts': pred_counts.tolist(),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp),
            'threshold': float(threshold),
            'threshold_source': threshold_source,
            'threshold_percentile': float(threshold_percentile),
            'score': score_name,
            **val_metrics,
            'train_score_mean': float(np.mean(train_scores)),
            'val_score_mean': float(np.mean(val_scores)),
            'test_score_mean': float(np.mean(test_scores)),
        }
        metrics_path = os.path.join(folder_path, 'forecast_anomaly_metrics.csv')
        with open(metrics_path, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_row))
            writer.writeheader()
            writer.writerow(metric_row)

        if sweep_rows:
            sweep_path = os.path.join(folder_path, 'threshold_sweep.csv')
            with open(sweep_path, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
                writer.writeheader()
                writer.writerows(sweep_rows)

        np.save(os.path.join(folder_path, 'test_scores.npy'), test_scores)
        np.save(os.path.join(folder_path, 'test_labels.npy'), gt)
        return
