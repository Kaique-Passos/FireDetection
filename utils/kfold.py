import os
import json
import shutil
import hashlib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset
from datasets import available_datasets
from .models import models_conf
from .training import get_model, train_model, test_model, save_history


def make_folds(labels, k=10, seed=666):
    """Stratified K-fold splits. Same labels + seed always yield the same folds,
    so every model is trained and validated on exactly the same partitions."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))
# end make_folds

def evaluate(model, dataset, images, batch_size=32, use_cuda=True):
    """Metrics plus the per-image predictions. `images` holds the rows of the
    dataset CSV in the same order as `dataset` (the test loader does not
    shuffle), so each prediction can be traced back to its image."""
    y_true, y_prob, _ = test_model(model, dataset, batch_size=batch_size, use_cuda=use_cuda)
    y_pred = y_prob.argmax(axis=1)
    # positive class: fire (idx 1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(roc_auc_score(y_true, y_prob[:, 1])),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

    predictions = images[['folder_path', 'image_id', 'class']].reset_index(drop=True)
    predictions['y_true'] = y_true
    predictions['y_pred'] = y_pred
    predictions['prob_fire'] = y_prob[:, 1]
    predictions['correct'] = (y_true == y_pred).astype(int)

    return metrics, predictions
# end evaluate

def save_history_plots(history, folder_path):
    for key, title in [('acc', 'accuracy'), ('loss', 'loss')]:
        fig = plt.figure()
        plt.plot(history[key])
        plt.plot(history['val_' + key])
        plt.title('Model {}'.format(title))
        plt.ylabel(title.capitalize())
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        fig.savefig(os.path.join(folder_path, '{}.png'.format(title)))
        plt.close(fig)
# end save_plots

IN_PROGRESS_FOLDER = '_em_andamento'
RUN_CONFIG_FILE = 'config.json'

def now_str(timezone='America/Sao_Paulo'):
    return datetime.now(ZoneInfo(timezone)).strftime('%Y-%m-%d_%H-%M-%S')
# end now_str

def start_run(history_root, base_model, run_config):
    """Folder where the run is trained: <history_root>/<model>/_em_andamento.

    An interrupted run with the same configuration is resumed; a different
    configuration raises an error instead of mixing results."""
    run_path = os.path.join(history_root, base_model, IN_PROGRESS_FOLDER)
    config_path = os.path.join(run_path, RUN_CONFIG_FILE)

    if os.path.exists(config_path):
        with open(config_path) as f:
            saved_config = json.load(f)
        saved_params = {c: v for c, v in saved_config.items() if c != 'started_at'}

        if saved_params != run_config:
            raise ValueError('There is an unfinished run with another configuration at {}.\n'
                             'Saved: {}\nCurrent: {}\nUse the same configuration to resume it '
                             'or delete that folder to start over.'.format(
                                run_path, saved_params, run_config))
        print('Resuming unfinished run started at', saved_config['started_at'])
    else:
        os.makedirs(run_path, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(dict(run_config, started_at=now_str()), f, indent=2)

    return run_path
# end start_run

def finish_run(history_root, base_model):
    """Move the finished run to <history_root>/<model>/<date_time>."""
    run_path = os.path.join(history_root, base_model, IN_PROGRESS_FOLDER)
    config_path = os.path.join(run_path, RUN_CONFIG_FILE)
    finished_at = now_str()

    with open(config_path) as f:
        run_config = json.load(f)
    run_config['finished_at'] = finished_at
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)

    final_path = os.path.join(history_root, base_model, finished_at)
    shutil.move(run_path, final_path)
    print('Run saved at', final_path)

    return final_path
# end finish_run

def list_runs(history_root, base_model):
    """Finished runs of a model, oldest first."""
    model_path = os.path.join(history_root, base_model)
    if not os.path.isdir(model_path):
        return []

    return sorted(os.path.join(model_path, d) for d in os.listdir(model_path)
                  if d != IN_PROGRESS_FOLDER
                  and os.path.exists(os.path.join(model_path, d, 'kfold_results.csv')))
# end list_runs

def run_kfold(base_model, folder_path, dataset_name='firenet', k=10, epochs=100,
              batch_size=32, seed=666, test_dataset_name='firenet_test',
              preload_data=True, use_cuda=None, save_models=False, save_plots=False):
    """Train `base_model` with stratified K-fold cross validation, saving the
    results at `folder_path`. By default only the metrics, the training
    history and the fold split are saved; `save_models` and `save_plots` also
    save each fold's best weights and accuracy/loss plots.

    The whole dataset (train + val rows of the CSV) is split in `k` folds; on
    each round one fold is held out for validation and the other k-1 are used
    for training. The best epoch's weights (by validation accuracy) are then
    evaluated on the held-out fold and on the independent test set.

    Folds already finished (fold_XX/metrics.json exists) are skipped, so the
    run can be resumed after a Colab disconnection.
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = models_conf[base_model]
    num_classes = available_datasets[dataset_name]['num_classes']
    base_dataset = available_datasets[dataset_name]['class']

    # purpose=None ignores the original train/val split and keeps every image
    train_full = base_dataset(transform=config['preprocess_train'], purpose=None,
                              preload=preload_data)
    if repr(config['preprocess_val']) == repr(config['preprocess_train']):
        val_full = train_full # same preprocessing, avoid loading the images twice
    else:
        val_full = base_dataset(transform=config['preprocess_val'], purpose=None,
                                preload=preload_data)

    test_data = None
    if test_dataset_name is not None:
        test_dataset = available_datasets[test_dataset_name]['class']
        test_data = test_dataset(transform=config['preprocess_test'], preload=preload_data)

    labels = train_full.data['class'].map(lambda c: train_full.labels[c]['idx']).values
    folds = make_folds(labels, k=k, seed=seed)

    os.makedirs(folder_path, exist_ok=True)
    print('K-fold ({} folds) results will be saved at {}'.format(k, folder_path))

    # keep a record of which image belongs to which validation fold
    folds_df = train_full.data[['folder_path', 'image_id', 'class']].copy()
    folds_df['fold'] = 0
    for fold, (_, val_idx) in enumerate(folds, start=1):
        folds_df.iloc[val_idx, folds_df.columns.get_loc('fold')] = fold
    folds_df.to_csv(os.path.join(folder_path, 'folds.csv'), index=False)

    # and of the independent test images, the same for every fold and model
    if test_data is not None:
        test_data.data[['folder_path', 'image_id', 'class']].to_csv(
            os.path.join(folder_path, 'test_images.csv'), index=False)

    results = []
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        fold_path = os.path.join(folder_path, 'fold_{:02d}'.format(fold))
        metrics_file = os.path.join(fold_path, 'metrics.json')

        if os.path.exists(metrics_file):
            print('Fold {:02d}/{:02d}: already done, skipping'.format(fold, k))
            with open(metrics_file) as f:
                results.append(json.load(f))
            continue

        os.makedirs(fold_path, exist_ok=True)
        print('Fold {:02d}/{:02d}: training on {} images, validating on {}'.format(
            fold, k, len(train_idx), len(val_idx)))

        # seed per fold, so a resumed run gives the same result as a full run
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        model, _ = get_model(base_model, num_classes=num_classes)
        criterion = config['criterion']
        opt_args = {'params': model.parameters()}
        opt_args.update(config['optimizer_params'])
        optimizer = config['optimizer'](**opt_args)
        scheduler = None

        if config['scheduler'] is not None:
            sched_args = {'optimizer': optimizer}
            sched_args.update(config['scheduler_params'])
            scheduler = config['scheduler'](**sched_args)

        train_data = Subset(train_full, train_idx)
        val_data = Subset(val_full, val_idx)

        with open(os.path.join(fold_path, 'training.log'), 'w') as f:
            with redirect_stdout(f):
                print(model)
                history, best_model, time_elapsed = train_model(model, criterion, optimizer,
                            train_data, val_data, epochs=epochs, batch_size=batch_size,
                            shuffle_dataset=True, scheduler=scheduler, use_cuda=use_cuda)

        save_history(history, file_path=os.path.join(fold_path, 'history.csv'))
        if save_models:
            torch.save(best_model, os.path.join(fold_path, config['model_path']))
        if save_plots:
            save_history_plots(history, fold_path)

        # evaluate the best epoch's weights
        model.load_state_dict(best_model)
        with open(os.path.join(fold_path, 'testing.log'), 'w') as f:
            with redirect_stdout(f):
                val_metrics, val_pred = evaluate(model, val_data, train_full.data.iloc[val_idx],
                                                 batch_size=batch_size, use_cuda=use_cuda)
                test_metrics = None
                if test_data is not None:
                    test_metrics, test_pred = evaluate(model, test_data, test_data.data,
                                                       batch_size=batch_size, use_cuda=use_cuda)

        # per-image predictions (saved before metrics.json, which marks the fold as done)
        val_pred.to_csv(os.path.join(fold_path, 'predictions_val.csv'), index=False)
        if test_data is not None:
            test_pred.to_csv(os.path.join(fold_path, 'predictions_test.csv'), index=False)

        row = {
            'fold': fold,
            'best_epoch': int(np.argmax(history['val_acc']) + 1),
            'train_time_s': float(time_elapsed)
        }
        row.update({'val_' + m: v for m, v in val_metrics.items()})
        if test_metrics is not None:
            row.update({'test_' + m: v for m, v in test_metrics.items()})

        with open(metrics_file, 'w') as f:
            json.dump(row, f, indent=2)
        results.append(row)

        summary_line = 'Fold {:02d}/{:02d} done in {:.0f}m {:.0f}s | val acc: {:.4f}'.format(
            fold, k, time_elapsed // 60, time_elapsed % 60, row['val_accuracy'])
        if test_metrics is not None:
            summary_line += ' | test acc: {:.4f}'.format(row['test_accuracy'])
        print(summary_line)

        # free GPU memory before the next fold
        del model, optimizer, scheduler, best_model
        if use_cuda:
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results).sort_values('fold').reset_index(drop=True)
    results_df.to_csv(os.path.join(folder_path, 'kfold_results.csv'), index=False)

    summary = results_df.drop(columns=['fold']).agg(['mean', 'std']).T
    summary.to_csv(os.path.join(folder_path, 'kfold_summary.csv'))

    return results_df, summary
# end run_kfold

def load_kfold_results(folder_path):
    return pd.read_csv(os.path.join(folder_path, 'kfold_results.csv'))
# end load_kfold_results

def file_hash(folder_path, file_name):
    """SHA-256 of a run's file: equal hashes mean identical files."""
    with open(os.path.join(folder_path, file_name), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()
# end file_hash

def check_same_file(run_paths, file_name):
    """Check that every run saved exactly the same `file_name` table. A run
    without the file (older runs) is reported as not matching."""
    reference = None
    report = []

    for path in run_paths:
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            report.append({'run': path, 'images': None, 'sha256': None,
                           'same_as_first': False})
            continue

        df = pd.read_csv(file_path)
        if reference is None:
            reference = df
        report.append({
            'run': path,
            'images': len(df),
            'sha256': file_hash(path, file_name),
            'same_as_first': df.equals(reference)
        })

    report = pd.DataFrame(report)
    return bool(report['same_as_first'].all()), report
# end check_same_file

def check_same_folds(run_paths):
    """Same image -> validation fold assignment in every run (folds.csv)."""
    return check_same_file(run_paths, 'folds.csv')
# end check_same_folds

def check_same_test_images(run_paths):
    """Same independent test images in every run (test_images.csv)."""
    return check_same_file(run_paths, 'test_images.csv')
# end check_same_test_images

def compare_kfold(results_a, results_b, name_a='A', name_b='B', metrics=None):
    """Mean ± std of each metric for both models, plus paired tests over the
    folds (both models used the same folds, so the samples are paired)."""
    if metrics is None:
        # rates only, the confusion matrix counts are left out
        metrics = [c for c in results_a.columns
                   if (c.startswith('val_') or c.startswith('test_'))
                   and c.split('_', 1)[1] not in ('tp', 'fp', 'tn', 'fn')]

    a = results_a.sort_values('fold').reset_index(drop=True)
    b = results_b.sort_values('fold').reset_index(drop=True)
    rows = []

    for m in metrics:
        diff = a[m] - b[m]
        row = {
            'metric': m,
            name_a: '{:.4f} ± {:.4f}'.format(a[m].mean(), a[m].std()),
            name_b: '{:.4f} ± {:.4f}'.format(b[m].mean(), b[m].std()),
            'diff_mean': diff.mean(),
            't_test_p': np.nan,
            'wilcoxon_p': np.nan
        }

        if np.any(diff != 0):
            row['t_test_p'] = stats.ttest_rel(a[m], b[m]).pvalue
            row['wilcoxon_p'] = stats.wilcoxon(a[m], b[m]).pvalue

        rows.append(row)

    return pd.DataFrame(rows)
# end compare_kfold
