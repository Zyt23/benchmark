from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

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
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
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
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        metric_path = './results/' + setting + '/'
        if not os.path.exists(metric_path):
            os.makedirs(metric_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduction='none')

        def _align_outputs(batch_x, outputs):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.shape[1] != batch_x.shape[1]:
                outputs = outputs[:, -batch_x.shape[1]:, :]
            if outputs.shape[-1] != batch_x.shape[-1]:
                outputs = outputs[:, :, -batch_x.shape[-1]:]
            return outputs

        def _collect_energy(loader):
            point_energy = []
            window_energy = []
            point_labels = []
            window_labels = []
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(loader):
                    batch_x = batch_x.float().to(self.device)
                    outputs = self.model(batch_x, None, None, None)
                    outputs = _align_outputs(batch_x, outputs)
                    score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
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

        level = getattr(self.args, 'anomaly_level', 'point')
        threshold_source = getattr(self.args, 'anomaly_threshold_source', 'combined')
        threshold_percentile = float(getattr(self.args, 'anomaly_threshold_percentile', 99.0))

        if level == 'window':
            train_energy = train_window_energy
            val_energy = val_window_energy
            test_energy = test_window_energy
            gt = test_window_labels
        else:
            train_energy = train_point_energy
            val_energy = val_point_energy
            test_energy = test_point_energy
            gt = test_point_labels

        if threshold_source == 'val':
            threshold = np.percentile(val_energy, threshold_percentile)
        elif threshold_source == 'train':
            threshold = np.percentile(train_energy, threshold_percentile)
        else:
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)
        print("Threshold source: {}, percentile: {}, level: {}".format(
            threshold_source, threshold_percentile, level))

        pred = (test_energy > threshold).astype(int)
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
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
        true_counts = np.bincount(gt.astype(int), minlength=2)
        pred_counts = np.bincount(pred.astype(int), minlength=2)
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
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

        with open(os.path.join(metric_path, 'anomaly_metrics.csv'), 'w') as f:
            f.write('setting,accuracy,precision,recall,f1,true_counts,pred_counts,TN,FP,FN,TP,threshold,threshold_source,threshold_percentile,level\n')
            f.write('{},{:.10f},{:.10f},{:.10f},{:.10f},"{}","{}",{},{},{},{},{:.10f},{},{:.4f},{}\n'.format(
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
                level,
            ))
        return
