import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from collections import deque
from river import cluster, stream, metrics

# function to return model names from a list of models
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

# main function for benchmark clustering
def benchmark_clustering(models, 
                         metric = metrics.cluster.Silhouette(), 
                         dataset_name = "powersupply", 
                         n_samples = 10000,
                         save = True):
    """Internal benchmark for clustering algorithms
    
    
    """
    
    # load dataset, check final feature and create stream
    data = pd.read_csv(dataset_name+".csv", nrows=n_samples)
    if data.columns[-1] == 'class':
        features = data.columns[:-2]
        data_stream = stream.iter_pandas(X=data[features], y=data["class"])
    else:
        features = data.columns # no ground truth
        data_stream = stream.iter_pandas(X=data[features])

    if (('river.metrics.cluster' not in metric.__module__) and (data.columns[-1] != 'class')):
        raise Exception('The dataset and metric used are not compatible. \n' 
                        'When the metric taken into acount is external, \n ground truth for the clustering problem must be provided.')
    
    # metric's name
    metric_name = str(metric.__module__).split('.')[-1]
    
    # extract number of models
    n_models = len(models)
    
    # extract model names
    models_names = [namestr(models[i], globals())[0] for i in range(n_models)]
    models_names_concat = '_'.join(models_names)
    
    # initiate metric, time and score array
    metric_array = [metric for _ in range(n_models)]
    time_array = np.zeros(n_models)
    score_array = [deque() for _ in range(n_models)]

    # loop function to update all models
    for i, (x, y_true) in enumerate(data_stream):
        for nth_model in range(n_models):
            start = time.perf_counter()
            models[nth_model].learn_one(x)
            y_pred = models[nth_model].predict_one(x)
            time_array[nth_model] += (time.perf_counter() - start)
            
            # update as suitable for internal metrics and external metrics
            if 'river.metrics.cluster' in metric.__module__:
                metric_array[nth_model].update(x=x, y_pred=y_pred, centers=models[nth_model].centers)
            else:
                metric_array[nth_model].update(y_true=y_true, y_pred=y_pred)
                
            if i == 0 or (i+1) % int(.01 * n_samples) == 0:
                score_array[nth_model].append((metric_array[nth_model].get(), models[nth_model]._raw_memory_usage, time_array[nth_model]))

    # form and save df
    
    df = pd.DataFrame(index=np.arange(0, n_samples + 1, int(0.01 * n_samples)))
    for i in range(n_models):
        df[models_names[i] + "_metric"] = [t[0] for t in score_array[i]]
        df[models_names[i] + "_memory"] = [t[1] for t in score_array[i]]
        df[models_names[i] + "_time"] = [t[2] for t in score_array[i]]
    
    if save:
        df.to_csv(f'{models_names_concat}_cluster_benchmark_with_metric_{metric_name}.csv')
    
    df
    
    # plot
    
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16,10))
    fig.suptitle(f'Models: {models_names} | Dataset name: {dataset_name} | n_samples: {n_samples}')

    
    # first subplot
    for i in range(n_models):
        axes[0].plot(df.index, df.iloc[:, 3*i], label = models_names[i])
    axes[0].set_ylabel(metric_name)
    axes[0].grid(linestyle=':')
    axes[0].legend()
    
    # second subplot
    for i in range(n_models):
        axes[1].plot(df.index, df.iloc[:, 3*i+1], label=models_names[i])
    axes[1].set_ylabel('Time (s)')
    axes[1].grid(linestyle=':')
    axes[1].legend()
    
    # third subplot
    for i in range(n_models):
        axes[2].plot(df.index, df.iloc[:, 3*i+2], label=models_names[i])
    axes[2].set_ylabel('Memory (bytes)')
    axes[2].grid(linestyle=':')
    axes[2].legend()
    
    fig.tight_layout()
    if save:
        plt.savefig(f'Models: {models_names} | Dataset name: {dataset_name} | n_samples: {n_samples} | Metric: {metric_name}.png')
    