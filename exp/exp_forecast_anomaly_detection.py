from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
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
        mse_scores = []
        mae_scores = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_label = batch
                outputs, true = self._forward_forecast(batch_x, batch_y, batch_x_mark, batch_y_mark)
                err = outputs - true
                mse = torch.mean(err ** 2, dim=(1, 2)).detach().cpu().numpy()
                mae = torch.mean(torch.abs(err), dim=(1, 2)).detach().cpu().numpy()
                mse_scores.append(mse)
                mae_scores.append(mae)
                labels.append(batch_label.detach().cpu().numpy().reshape(-1))
        return (
            np.concatenate(mse_scores, axis=0),
            np.concatenate(mae_scores, axis=0),
            np.concatenate(labels, axis=0).astype(int),
        )

    @staticmethod
    def _best_f1_threshold(scores, labels):
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores).astype(float)
        if labels.max(initial=0) == 0 or scores.size == 0:
            return None, None

        percentiles = np.concatenate([
            np.linspace(50.0, 95.0, 46),
            np.linspace(95.5, 99.5, 9),
            np.asarray([99.7, 99.9]),
        ])
        candidates = np.unique(np.percentile(scores, percentiles))
        best = None
        for threshold in candidates:
            pred = (scores > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, pred, average='binary', zero_division=0)
            acc = accuracy_score(labels, pred)
            current = (float(f1), float(acc), float(recall), float(precision), float(threshold))
            if best is None or current[:4] > best[:4]:
                best = current
        if best is None:
            return None, None
        f1, acc, recall, precision, threshold = best
        return threshold, {
            'val_accuracy': acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
        }

    def _select_threshold(self, train_scores, val_scores, threshold_scores, threshold_labels):
        source = getattr(self.args, 'anomaly_threshold_source', 'val')
        percentile = float(getattr(self.args, 'anomaly_threshold_percentile', 99.0))

        if source == 'val_mixed_best_f1':
            threshold, val_metrics = self._best_f1_threshold(threshold_scores, threshold_labels)
            if threshold is not None:
                return float(threshold), source, percentile, val_metrics
            source = 'val'

        if source == 'train':
            return float(np.percentile(train_scores, percentile)), source, percentile, {}
        if source == 'combined':
            combined = np.concatenate([train_scores, val_scores], axis=0)
            return float(np.percentile(combined, percentile)), source, percentile, {}
        return float(np.percentile(val_scores, percentile)), 'val', percentile, {}

    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        threshold_data, threshold_loader = self._get_data(flag='threshold')
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                map_location=self.device,
            ))

        train_mse, train_mae, _ = self._collect_scores(train_loader)
        val_mse, val_mae, _ = self._collect_scores(val_loader)
        thre_mse, thre_mae, thre_labels = self._collect_scores(threshold_loader)
        test_mse, test_mae, gt = self._collect_scores(test_loader)

        score_name = str(getattr(self.args, 'forecast_anomaly_score', 'mse')).lower()
        if score_name == 'mae':
            train_scores, val_scores, threshold_scores, test_scores = train_mae, val_mae, thre_mae, test_mae
        else:
            score_name = 'mse'
            train_scores, val_scores, threshold_scores, test_scores = train_mse, val_mse, thre_mse, test_mse

        threshold, threshold_source, threshold_percentile, val_metrics = self._select_threshold(
            train_scores, val_scores, threshold_scores, thre_labels)

        pred = (test_scores > threshold).astype(int)
        gt = np.asarray(gt).astype(int)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
        true_counts = np.bincount(gt.astype(int), minlength=2)
        pred_counts = np.bincount(pred.astype(int), minlength=2)

        print("Threshold: {}".format(threshold))
        print("Threshold source: {}, percentile: {}, score: {}".format(
            threshold_source, threshold_percentile, score_name))
        if val_metrics:
            print("Threshold validation metrics: {}".format(val_metrics))
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
            accuracy, precision, recall, f_score))
        print("true_counts: {}, pred_counts: {}, TN/FP/FN/TP: {}/{}/{}/{}".format(
            true_counts.tolist(), pred_counts.tolist(), int(tn), int(fp), int(fn), int(tp)))

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, 'forecast_anomaly_metrics.csv'), 'w', encoding='utf-8') as f:
            f.write(
                'setting,accuracy,precision,recall,f1,true_counts,pred_counts,'
                'TN,FP,FN,TP,threshold,threshold_source,threshold_percentile,score,'
                'val_accuracy,val_precision,val_recall,val_f1,train_score_mean,val_score_mean,test_score_mean\n'
            )
            f.write('{},{:.10f},{:.10f},{:.10f},{:.10f},"{}","{}",{},{},{},{},{:.10f},{},{:.4f},{},{},{},{},{},{:.10f},{:.10f},{:.10f}\n'.format(
                setting,
                float(accuracy),
                float(precision),
                float(recall),
                float(f_score),
                true_counts.tolist(),
                pred_counts.tolist(),
                int(tn),
                int(fp),
                int(fn),
                int(tp),
                float(threshold),
                threshold_source,
                threshold_percentile,
                score_name,
                val_metrics.get('val_accuracy', ''),
                val_metrics.get('val_precision', ''),
                val_metrics.get('val_recall', ''),
                val_metrics.get('val_f1', ''),
                float(np.mean(train_scores)),
                float(np.mean(val_scores)),
                float(np.mean(test_scores)),
            ))

        np.save(os.path.join(folder_path, 'test_scores.npy'), test_scores)
        np.save(os.path.join(folder_path, 'test_labels.npy'), gt)
        return
