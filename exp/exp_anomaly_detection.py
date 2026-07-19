from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import csv
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    @staticmethod
    def _align_tensor(batch_x, tensor):
        if tensor.shape[1] != batch_x.shape[1]:
            tensor = tensor[:, -batch_x.shape[1]:, :]
        if tensor.shape[-1] != batch_x.shape[-1]:
            tensor = tensor[:, :, -batch_x.shape[-1]:]
        return tensor

    @staticmethod
    def _association_kl(p, q):
        values = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
        return torch.mean(torch.sum(values, dim=-1), dim=1)

    def _anomaly_transformer_objectives(self, output, target):
        reconstruction, series, prior = output[0], output[1], output[2]
        reconstruction = self._align_tensor(target, reconstruction)
        rec_loss = nn.functional.mse_loss(reconstruction, target)
        series_loss = 0.0
        prior_loss = 0.0
        for series_u, prior_u in zip(series, prior):
            prior_norm = prior_u / (prior_u.sum(dim=-1, keepdim=True) + 1e-12)
            series_loss = series_loss + torch.mean(
                self._association_kl(series_u, prior_norm.detach())
                + self._association_kl(prior_norm.detach(), series_u)
            )
            prior_loss = prior_loss + torch.mean(
                self._association_kl(prior_norm, series_u.detach())
                + self._association_kl(series_u.detach(), prior_norm)
            )
        layer_count = max(1, len(prior))
        series_loss = series_loss / layer_count
        prior_loss = prior_loss / layer_count
        association_k = float(getattr(self.args, 'anomaly_association_k', 3.0))
        return rec_loss - association_k * series_loss, rec_loss + association_k * prior_loss

    def _training_loss(self, output, target, epoch_idx):
        model_name = str(self.args.model)
        mse = nn.MSELoss(reduction='none')

        if model_name == 'USAD' and isinstance(output, (tuple, list)) and len(output) >= 3:
            ae1, ae2, ae2ae1 = [self._align_tensor(target, item) for item in output[:3]]
            n = float(epoch_idx + 1)
            loss1 = (1.0 / n) * mse(ae1, target) + (1.0 - 1.0 / n) * mse(ae2ae1, target)
            loss2 = (1.0 / n) * mse(ae2, target) - (1.0 - 1.0 / n) * mse(ae2ae1, target)
            return (loss1 + loss2).mean()

        if model_name == 'TranAD' and isinstance(output, (tuple, list)) and len(output) >= 2:
            phase1, phase2 = [self._align_tensor(target, item) for item in output[:2]]
            n = float(epoch_idx + 1)
            return ((1.0 / n) * mse(phase1, target) + (1.0 - 1.0 / n) * mse(phase2, target)).mean()

        if model_name == 'OmniAnomaly' and isinstance(output, (tuple, list)) and len(output) >= 3:
            reconstruction = self._align_tensor(target, output[0])
            mu, logvar = output[1], output[2]
            reconstruction_loss = mse(reconstruction, target).mean()
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            beta = float(getattr(self.args, 'omni_beta', 0.001))
            return reconstruction_loss + beta * kld

        reconstruction = output[0] if isinstance(output, (tuple, list)) else output
        reconstruction = self._align_tensor(target, reconstruction)
        return mse(reconstruction, target).mean()

    def _score_tensor(self, output, target):
        model_name = str(self.args.model)
        mse = nn.MSELoss(reduction='none')

        if model_name == 'AnomalyTransformer' and isinstance(output, (tuple, list)) and len(output) >= 4:
            reconstruction, series, prior = output[0], output[1], output[2]
            reconstruction = self._align_tensor(target, reconstruction)
            reconstruction_error = torch.mean(mse(reconstruction, target), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            temperature = float(getattr(self.args, 'anomaly_temperature', 50.0))
            for series_u, prior_u in zip(series, prior):
                prior_norm = prior_u / (prior_u.sum(dim=-1, keepdim=True) + 1e-12)
                series_loss = series_loss + self._association_kl(
                    series_u, prior_norm.detach()) * temperature
                prior_loss = prior_loss + self._association_kl(
                    prior_norm, series_u.detach()) * temperature
            association_weight = torch.softmax(-(series_loss + prior_loss), dim=-1)
            return association_weight * reconstruction_error

        if model_name == 'USAD' and isinstance(output, (tuple, list)) and len(output) >= 3:
            ae1 = self._align_tensor(target, output[0])
            ae2ae1 = self._align_tensor(target, output[2])
            return torch.mean(0.1 * mse(ae1, target) + 0.9 * mse(ae2ae1, target), dim=-1)

        if model_name == 'TranAD' and isinstance(output, (tuple, list)) and len(output) >= 2:
            reconstruction = self._align_tensor(target, output[1])
        else:
            reconstruction = output[0] if isinstance(output, (tuple, list)) else output
            reconstruction = self._align_tensor(target, reconstruction)
        return torch.mean(mse(reconstruction, target), dim=-1)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_loss2 = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)
                if str(self.args.model) == 'AnomalyTransformer' and isinstance(outputs, (tuple, list)) and len(outputs) >= 4:
                    loss1, loss2 = self._anomaly_transformer_objectives(outputs, batch_x)
                    total_loss.append(loss1.item())
                    total_loss2.append(loss2.item())
                else:
                    loss = self._score_tensor(outputs, batch_x).mean()
                    total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        if total_loss2:
            return total_loss, np.average(total_loss2)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        dual_best = None
        dual_counter = 0

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                anomaly_transformer = (
                    str(self.args.model) == 'AnomalyTransformer'
                    and isinstance(outputs, (tuple, list))
                    and len(outputs) >= 4
                )
                if anomaly_transformer:
                    loss, loss2 = self._anomaly_transformer_objectives(outputs, batch_x)
                else:
                    loss = self._training_loss(outputs, batch_x, epoch)
                    loss2 = None
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if loss2 is not None:
                    # Minimax association learning from the official Anomaly
                    # Transformer training procedure.
                    loss.backward(retain_graph=True)
                    loss2.backward()
                else:
                    loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_result = self.vali(vali_data, vali_loader, criterion)
            if isinstance(vali_result, tuple):
                vali_loss, vali_loss2 = vali_result
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss1: {3:.7f} Vali Loss2: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, vali_loss2))
                current = (float(vali_loss), float(vali_loss2))
                if dual_best is None or (current[0] < dual_best[0] and current[1] < dual_best[1]):
                    dual_best = current
                    dual_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
                    print('Dual validation objectives improved. Saving model ...')
                else:
                    dual_counter += 1
                    print('EarlyStopping counter: {} out of {}'.format(dual_counter, self.args.patience))
                    if dual_counter >= self.args.patience:
                        print("Early stopping")
                        break
            else:
                vali_loss = vali_result
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    @staticmethod
    def _best_f1_threshold(scores, labels, directions=(1.0, -1.0)):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if scores.size == 0 or np.unique(labels).size < 2:
            return None, 1.0, {}

        percentiles = np.linspace(0.0, 100.0, 1001)
        best = None
        for direction in directions:
            decision_scores = scores * float(direction)
            candidates = np.unique(np.percentile(decision_scores, percentiles))
            for threshold in candidates:
                pred = (decision_scores > threshold).astype(np.int64)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, pred, average='binary', zero_division=0)
                accuracy = accuracy_score(labels, pred)
                current = (float(f1), float(recall), float(precision), float(accuracy), -float(threshold))
                if best is None or current > best[0]:
                    best = (current, float(threshold), float(direction))

        metrics, threshold, direction = best
        return threshold, direction, {
            'threshold_val_f1': metrics[0],
            'threshold_val_recall': metrics[1],
            'threshold_val_precision': metrics[2],
            'threshold_val_accuracy': metrics[3],
        }

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        threshold_source = getattr(self.args, 'anomaly_threshold_source', 'val')
        threshold_data = threshold_loader = None
        if threshold_source == 'val_mixed_best_f1':
            threshold_data, threshold_loader = self._get_data(flag='threshold')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'),
                map_location=self.device,
            ))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        metric_path = './results/' + setting + '/'
        if not os.path.exists(metric_path):
            os.makedirs(metric_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduction='none')

        def _collect_energy(loader):
            point_energy = []
            window_energy = []
            point_labels = []
            window_labels = []
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(loader):
                    batch_x = batch_x.float().to(self.device)
                    outputs = self.model(batch_x, None, None, None)
                    score = self._score_tensor(outputs, batch_x)
                    score_np = score.detach().cpu().numpy()
                    point_energy.append(score_np.reshape(-1))
                    window_energy.append(score_np.mean(axis=1))

                    labels_np = batch_y.detach().cpu().numpy() if torch.is_tensor(batch_y) else np.asarray(batch_y)
                    point_labels.append(labels_np.reshape(-1))
                    if labels_np.ndim <= 1:
                        win_label = labels_np.reshape(-1)
                    else:
                        axes = tuple(range(1, labels_np.ndim))
                        win_label = labels_np.max(axis=axes)
                    window_labels.append(win_label.reshape(-1))

            return (
                np.concatenate(point_energy, axis=0),
                np.concatenate(window_energy, axis=0),
                np.concatenate(point_labels, axis=0).astype(int),
                np.concatenate(window_labels, axis=0).astype(int),
            )

        train_point_energy, train_window_energy, _, _ = _collect_energy(train_loader)
        val_point_energy, val_window_energy, _, _ = _collect_energy(vali_loader)
        test_point_energy, test_window_energy, test_point_labels, test_window_labels = _collect_energy(test_loader)
        if threshold_loader is not None:
            threshold_point_energy, threshold_window_energy, threshold_point_labels, threshold_window_labels = _collect_energy(
                threshold_loader)
        else:
            threshold_point_energy = threshold_window_energy = np.asarray([], dtype=np.float64)
            threshold_point_labels = threshold_window_labels = np.asarray([], dtype=np.int64)

        level = getattr(self.args, 'anomaly_level', 'point')
        threshold_percentile = float(getattr(self.args, 'anomaly_threshold_percentile', 99.0))

        if level == 'window':
            train_energy = train_window_energy
            val_energy = val_window_energy
            test_energy = test_window_energy
            gt = test_window_labels
            threshold_energy = threshold_window_energy
            threshold_labels = threshold_window_labels
        else:
            train_energy = train_point_energy
            val_energy = val_point_energy
            test_energy = test_point_energy
            gt = test_point_labels
            threshold_energy = threshold_point_energy
            threshold_labels = threshold_point_labels

        threshold_val_metrics = {}
        requested_direction = str(getattr(self.args, 'anomaly_score_direction', 'auto')).lower()
        direction_factor = -1.0 if requested_direction == 'low' else 1.0
        score_direction = 'lower_error_is_anomaly' if direction_factor < 0 else 'higher_error_is_anomaly'
        if threshold_source == 'val_mixed_best_f1':
            if requested_direction == 'high':
                candidate_directions = (1.0,)
            elif requested_direction == 'low':
                candidate_directions = (-1.0,)
            else:
                candidate_directions = (1.0, -1.0)
            threshold, direction_factor, threshold_val_metrics = self._best_f1_threshold(
                threshold_energy, threshold_labels, directions=candidate_directions)
            if threshold is None:
                print('Mixed validation threshold is unavailable; falling back to validation percentile.')
                threshold_source = 'val'
                direction_factor = 1.0
                threshold = np.percentile(val_energy, threshold_percentile)
            score_direction = 'lower_error_is_anomaly' if direction_factor < 0 else 'higher_error_is_anomaly'
        elif threshold_source == 'val':
            threshold = np.percentile(direction_factor * val_energy, threshold_percentile)
        elif threshold_source == 'train':
            threshold = np.percentile(direction_factor * train_energy, threshold_percentile)
        else:
            # Do not use test scores to choose a threshold.  The historical
            # TSLib "combined" mode mixed train and test energy and therefore
            # leaked held-out information.  Keep backward compatibility while
            # using train+validation only.
            combined_energy = np.concatenate([train_energy, val_energy], axis=0)
            threshold = np.percentile(direction_factor * combined_energy, 100 - self.args.anomaly_ratio)
            threshold_source = 'train_val'
        print("Threshold :", threshold)
        print("Threshold source: {}, percentile: {}, level: {}".format(
            threshold_source, threshold_percentile, level))

        decision_test_energy = direction_factor * test_energy
        pred = (decision_test_energy > threshold).astype(int)
        gt = np.array(gt).astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        if level == 'point':
            gt, pred = adjustment(gt, pred)
            pred = np.array(pred)
            gt = np.array(gt)

        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        balanced_accuracy = balanced_accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            gt, pred, average='macro', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
        true_counts = np.bincount(gt.astype(int), minlength=2)
        pred_counts = np.bincount(pred.astype(int), minlength=2)
        if np.unique(gt).size >= 2:
            raw_roc_auc = float(roc_auc_score(gt, test_energy))
            raw_pr_auc = float(average_precision_score(gt, test_energy))
            roc_auc = float(roc_auc_score(gt, decision_test_energy))
            pr_auc = float(average_precision_score(gt, decision_test_energy))
        else:
            raw_roc_auc = float('nan')
            raw_pr_auc = float('nan')
            roc_auc = float('nan')
            pr_auc = float('nan')
        normal_scores = test_energy[gt == 0]
        fault_scores = test_energy[gt == 1]

        def _stat(values, reducer):
            return float(reducer(values)) if values.size else float('nan')

        score_stats = {
            'normal_score_mean': _stat(normal_scores, np.mean),
            'fault_score_mean': _stat(fault_scores, np.mean),
            'normal_score_median': _stat(normal_scores, np.median),
            'fault_score_median': _stat(fault_scores, np.median),
            'normal_score_p95': _stat(normal_scores, lambda x: np.percentile(x, 95.0)),
            'fault_score_p95': _stat(fault_scores, lambda x: np.percentile(x, 95.0)),
        }
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        print("Balanced accuracy: {:0.4f}, Macro F1: {:0.4f}, ROC-AUC: {:0.4f}, PR-AUC: {:0.4f}".format(
            balanced_accuracy, macro_f1, roc_auc, pr_auc))
        print("Raw ROC-AUC: {:0.4f}, raw PR-AUC: {:0.4f}, direction: {}".format(
            raw_roc_auc, raw_pr_auc, score_direction))
        print("Score means normal/fault: {:.8f}/{:.8f}".format(
            score_stats['normal_score_mean'], score_stats['fault_score_mean']))
        print("true_counts: {}, pred_counts: {}, TN/FP/FN/TP: {}/{}/{}/{}".format(
            true_counts.tolist(), pred_counts.tolist(), int(tn), int(fp), int(fn), int(tp)))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()

        np.savez_compressed(
            os.path.join(metric_path, 'anomaly_scores.npz'),
            train_scores=np.asarray(train_energy, dtype=np.float32),
            val_scores=np.asarray(val_energy, dtype=np.float32),
            threshold_scores=np.asarray(threshold_energy, dtype=np.float32),
            threshold_labels=np.asarray(threshold_labels, dtype=np.int8),
            test_scores=np.asarray(test_energy, dtype=np.float32),
            test_decision_scores=np.asarray(decision_test_energy, dtype=np.float32),
            test_labels=np.asarray(gt, dtype=np.int8),
            test_predictions=np.asarray(pred, dtype=np.int8),
        )

        row = {
            'setting': setting,
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f_score),
            'macro_f1': float(macro_f1),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'raw_roc_auc': raw_roc_auc,
            'raw_pr_auc': raw_pr_auc,
            'true_counts': true_counts.tolist(),
            'pred_counts': pred_counts.tolist(),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp),
            'threshold': float(threshold),
            'threshold_source': threshold_source,
            'threshold_percentile': threshold_percentile,
            'level': level,
            'score_direction': score_direction,
            **score_stats,
            'threshold_val_accuracy': threshold_val_metrics.get('threshold_val_accuracy', float('nan')),
            'threshold_val_precision': threshold_val_metrics.get('threshold_val_precision', float('nan')),
            'threshold_val_recall': threshold_val_metrics.get('threshold_val_recall', float('nan')),
            'threshold_val_f1': threshold_val_metrics.get('threshold_val_f1', float('nan')),
        }
        with open(os.path.join(metric_path, 'anomaly_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return
