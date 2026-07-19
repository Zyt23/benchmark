from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, cal_precision_recall_f1
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        # Model construction must not inspect TEST, even for shape metadata.
        # QAR compact caches use a fixed sequence length, so TRAIN/VAL fully
        # determine the input shape required at deployment time.
        self.args.seq_len = max(train_data.max_seq_len, vali_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _labels_from_dataset(self, data_set):
        """Return integer labels for the current split, when the dataset exposes them."""
        if hasattr(data_set, '_compact_labels') and hasattr(data_set, 'samples'):
            sample_ids = np.asarray(data_set.samples, dtype=np.int64)
            return np.asarray(data_set._compact_labels[sample_ids], dtype=np.int64)

        if hasattr(data_set, 'samples'):
            labels = []
            for sample in data_set.samples:
                if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    labels.append(int(sample[1]))
            if labels:
                return np.asarray(labels, dtype=np.int64)

        if hasattr(data_set, 'labels_df') and hasattr(data_set, 'all_IDs'):
            values = data_set.labels_df.loc[data_set.all_IDs].values.reshape(-1)
            return values.astype(np.int64)

        return None

    def _select_criterion(self, train_data=None):
        if getattr(self.args, 'class_weight', 'none') == 'balanced' and train_data is not None:
            labels = self._labels_from_dataset(train_data)
            if labels is not None and labels.size:
                counts = np.bincount(labels, minlength=self.args.num_class).astype(np.float32)
                weights = np.zeros(self.args.num_class, dtype=np.float32)
                nonzero = counts > 0
                weights[nonzero] = counts.sum() / (nonzero.sum() * counts[nonzero])
                print('Using balanced class weights: counts={} weights={}'.format(
                    counts.astype(int).tolist(), weights.tolist()))
                weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
                return nn.CrossEntropyLoss(weight=weight_tensor)
            print('class_weight=balanced requested, but labels could not be inferred; using unweighted loss')
        return nn.CrossEntropyLoss()

    def _classification_metrics(self, predictions, trues, data_set=None):
        accuracy = cal_accuracy(predictions, trues)
        cls_names = getattr(data_set, 'class_names', None)
        report_str, report_dict = cal_precision_recall_f1(
            predictions, trues, class_names=cls_names)
        return {
            'accuracy': accuracy,
            'macro_f1': report_dict['macro avg']['f1-score'],
            'weighted_f1': report_dict['weighted avg']['f1-score'],
            'report_str': report_str,
            'report_dict': report_dict,
        }

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().view(-1))
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        metrics = self._classification_metrics(predictions, trues, vali_data)

        # 每类 precision/recall/F1（紧凑单行，便于每个 epoch 查看）
        try:
            report_dict = metrics['report_dict']
            per_class = []
            for key in report_dict:
                if key in ('accuracy', 'macro avg', 'weighted avg'):
                    continue
                per_class.append("{}=[P {:.3f}/R {:.3f}/F1 {:.3f}]".format(
                    key, report_dict[key]['precision'], report_dict[key]['recall'], report_dict[key]['f1-score']))
            print("\tper-class {}".format(" ".join(per_class)) +
                  " | macro-F1 {:.3f} weighted-F1 {:.3f}".format(metrics['macro_f1'], metrics['weighted_f1']))
        except Exception as e:
            print("\tper-class metrics skipped:", e)

        self.model.train()
        return total_loss, metrics['accuracy'], metrics['macro_f1'], metrics['weighted_f1']

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        # Treat exact metric ties as "no improvement"; otherwise a plateau at
        # macro_f1=1.0 keeps rewriting the best checkpoint every epoch.
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=1e-12)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(train_data)
        early_stop_metric = getattr(self.args, 'early_stop_metric', 'accuracy')
        print('Early stopping metric: {}'.format(early_stop_metric))

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().view(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy, val_macro_f1, val_weighted_f1 = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali MacroF1: {5:.3f} Vali WeightedF1: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, val_macro_f1, val_weighted_f1))
            monitor_values = {
                'accuracy': val_accuracy,
                'macro_f1': val_macro_f1,
                'weighted_f1': val_weighted_f1,
            }
            if early_stop_metric == 'loss':
                early_stopping(vali_loss, self.model, path)
                monitor_value = vali_loss
            else:
                monitor_value = monitor_values[early_stop_metric]
                early_stopping(-monitor_value, self.model, path)
            print("Early stopping monitor {}: {:.6f}".format(early_stop_metric, monitor_value))
            # Optional per-epoch checkpoints. The best checkpoint is still saved
            # by EarlyStopping above; large multi-dataset sweeps can disable
            # these to avoid filling the server disk.
            if int(getattr(self.args, 'save_epoch_checkpoints', 1)):
                epoch_ckpt = os.path.join(path, 'checkpoint_epoch{:03d}_valacc{:.4f}_valmacro{:.4f}.pth'.format(
                    epoch + 1, val_accuracy, val_macro_f1))
                torch.save(self.model.state_dict(), epoch_ckpt)
                print("Saved per-epoch checkpoint: {}".format(epoch_ckpt))
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        metrics = self._classification_metrics(predictions, trues, test_data)

        # 每类 precision / recall / F1 + macro/weighted 平均（完整报告）
        report_str = metrics['report_str']
        report_dict = metrics['report_dict']
        accuracy = metrics['accuracy']
        true_counts = np.bincount(trues, minlength=self.args.num_class).astype(int)
        pred_counts = np.bincount(predictions, minlength=self.args.num_class).astype(int)
        confusion = np.zeros((self.args.num_class, self.args.num_class), dtype=int)
        for true_label, pred_label in zip(trues, predictions):
            if 0 <= true_label < self.args.num_class and 0 <= pred_label < self.args.num_class:
                confusion[int(true_label), int(pred_label)] += 1
        if self.args.num_class == 2:
            tn, fp, fn, tp = confusion.ravel().tolist()
        else:
            tn = fp = fn = tp = ''

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        print('true counts:{}'.format(true_counts.tolist()))
        print('pred counts:{}'.format(pred_counts.tolist()))
        print('confusion matrix:{}'.format(confusion.tolist()))
        if self.args.num_class == 2:
            print('TN:{} FP:{} FN:{} TP:{}'.format(tn, fp, fn, tp))
        print('\nClassification Report (precision / recall / f1-score per class):\n')
        print(report_str)
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}\n'.format(accuracy))
        f.write('macro F1:{}\n'.format(report_dict['macro avg']['f1-score']))
        f.write('weighted F1:{}\n'.format(report_dict['weighted avg']['f1-score']))
        f.write('true counts:{}\n'.format(true_counts.tolist()))
        f.write('pred counts:{}\n'.format(pred_counts.tolist()))
        f.write('confusion matrix:{}\n'.format(confusion.tolist()))
        if self.args.num_class == 2:
            f.write('TN:{}\nFP:{}\nFN:{}\nTP:{}\n'.format(tn, fp, fn, tp))
        f.write('\nClassification Report:\n')
        f.write(report_str)
        f.write('\n\n')
        f.close()
        return
