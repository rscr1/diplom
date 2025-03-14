import os

from clearml import Task
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from utils_func import get_mask, get_depth, bn_off


def save_metrics(losses, metrics, primary_metric_per_class, losses_path=None, metrics_path=None, dataframe_path=None):
    if losses_path:
        plt.figure(figsize=(10, 5))
        plt.plot(losses['train'], label='Loss (train)')
        plt.plot(losses['valid'], label='Loss (valid)')
        plt.grid()
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(losses_path)
        plt.close()

    if metrics_path:
        plt.figure(figsize=(10, 5))
        for metric_name, metric_values in metrics.items():
            plt.plot(metric_values['train'], label=f'{metric_name} (train)', linestyle='-')
            plt.plot(metric_values['valid'], label=f'{metric_name} (valid)', linestyle='--')
        plt.grid()
        plt.title('Training and Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(metrics_path)
        plt.close()

    if dataframe_path and primary_metric_per_class is not None:
        pd.DataFrame(
            {i: class_metrics for i, class_metrics in enumerate(primary_metric_per_class)}
        ).to_csv(dataframe_path, index=True)

def train_loop(
    model,
    dataloader,
    loss_func,
    optimizer,
    metrics,
    parallel_type,
    device,
    classes,
    early_stopping,
    is_segm_task,
    model_name,
    scheduler=None,
    epochs=10,
    checkpoints_path=None,
    losses_path=None,
    metrics_path=None,
    dataframe_path=None
):
    # Get base path for model checkpoints
    best_model_path = os.path.join(checkpoints_path, f'best_model')
    last_model_path = os.path.join(checkpoints_path, f'last_model')
    
    # Get ClearML task
    task = Task.current_task()
    
    losses = {
        'train': [],
        'valid': []
    }

    # Initialize metrics tracking for each metric
    metrics_values = {name: {'train': [], 'valid': []} for name in metrics.keys()}
    primary_metric_per_class = [[] for _ in range(classes - 1)] if is_segm_task else None
    
    if is_segm_task:
        best_metric = 0
        primary_metric = 'iou' if 'iou' in metrics else list(metrics.keys())[0]
    else:
        best_metric = float('inf')
        primary_metric = 'rmse' if 'rmse' in metrics else list(metrics.keys())[0]
    
    best_epoch = 0
    
    for epoch in range(epochs):
        for phase in ('train', 'valid'):
            total_loss = 0
            
            # Initialize metric tracking for this epoch
            if is_segm_task:
                total_metrics_per_class = {
                    name: torch.zeros(classes - 1, device=device)
                    for name in metrics.keys()
                }
            else:
                total_metrics = {name: 0 for name in metrics.keys()}
                
            for batch in tqdm(dataloader[phase]):
                inputs, labels = batch
                
                # Handle label types
                if is_segm_task:
                    labels = labels.type(torch.LongTensor)
                else:
                    labels = labels.type(torch.FloatTensor)
                    
                inputs, labels = inputs.to(device), labels.to(device)
                bn_off(model, phase, is_segm_task, model_name)

                if phase == 'train':
                    optimizer.zero_grad()
                    out = model(inputs)
                    
                    # Handle outputs based on task
                    if is_segm_task:
                        mask = get_mask(out)
                        loss = loss_func(mask, labels)
                    else:
                        depth = get_depth(out)
                        outputs = depth[:, 0, :, :]
                        loss = loss_func(outputs, labels)
                            
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        out = model(inputs)
                        
                        # Handle outputs based on task
                        if is_segm_task:
                            mask = get_mask(out)
                            loss = loss_func(mask, labels)
                        else:
                            depth = get_depth(out)
                            outputs = depth[:, 0, :, :]
                            loss = loss_func(outputs, labels)

                total_loss += loss.item()
                
                # Calculate all metrics
                if is_segm_task:
                    if 'iou' in set(metrics.keys()):
                        total_metrics_per_class['iou'] += torch.nan_to_num(metrics['iou'](mask.argmax(1), labels), 0)
                    if 'dice' in set(metrics.keys()):
                        total_metrics_per_class['dice'] += torch.nan_to_num(metrics['dice'](mask.argmax(1), labels)[1:], 0)
                else:
                    for name, m in metrics.items():
                        total_metrics[name] += m(outputs, labels)

            # Average loss
            total_loss /= len(dataloader[phase])
            losses[phase].append(total_loss)

            # Log loss to ClearML
            if task is not None:
                task.get_logger().report_scalar(
                    title="Loss",
                    series=f"{phase}",
                    value=total_loss,
                    iteration=epoch
                )

            # Calculate and log metrics
            if is_segm_task:
                # Process each metric
                for name in metrics.keys():
                    epoch_metric_per_class = total_metrics_per_class[name] / len(dataloader[phase])
                    epoch_metric_per_class = epoch_metric_per_class.detach().cpu().numpy()
                    epoch_metric = epoch_metric_per_class.mean()
                    metrics_values[name][phase].append(epoch_metric)
                    if name == primary_metric and phase == 'valid':
                        [primary_metric_per_class[i].append(epoch_metric_per_class[i]) for i in range(classes - 1)]
                    
                    # Log per-class metrics to ClearML
                    if task is not None:
                        for i, class_metric in enumerate(epoch_metric_per_class):
                            task.get_logger().report_scalar(
                                title=f"{name}/{phase}",
                                series=f"class_{i}",
                                value=class_metric,
                                iteration=epoch
                            )
                        
                        # Log average metric
                        task.get_logger().report_scalar(
                            title=f"Metrics/{phase}",
                            series=name,
                            value=epoch_metric,
                            iteration=epoch
                        )
                
                # Use primary metric for model selection
                epoch_metric = metrics_values[primary_metric][phase][-1]
            else:
                # Process each metric
                for name in metrics.keys():
                    epoch_metric = total_metrics[name] / len(dataloader[phase])
                    epoch_metric = epoch_metric.detach().cpu().numpy()
                    metrics_values[name][phase].append(epoch_metric)
                    
                    # Log all metrics to ClearML in one graph
                    if task is not None:
                        task.get_logger().report_scalar(
                            title=f"Metrics/{phase}",
                            series=name,
                            value=epoch_metric,
                            iteration=epoch
                        )
                
                # Use primary metric for model selection
                epoch_metric = metrics_values[primary_metric][phase][-1]

            if phase == 'valid':
                print('Epoch', epoch)
                # print('Train Loss:', losses['train'][-1])
                # print('Valid Loss:', losses['valid'][-1])
                for name in metrics.keys():
                    print(f'Train {name}:', round(metrics_values[name]['train'][-1].item(), 4))
                    print(f'Valid {name}:', round(metrics_values[name]['valid'][-1].item(), 4))
                
                # Save last checkpoint
                if parallel_type is None:
                    torch.save(model.state_dict(), last_model_path)
                else:
                    torch.save(model.module.state_dict(), last_model_path)
                print('Last checkpoint saved')
                
                # Check and save best checkpoint
                is_better = epoch_metric > best_metric if is_segm_task else epoch_metric < best_metric
                if is_better:
                    best_metric = epoch_metric
                    best_epoch = epoch
                    if parallel_type is None:
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        torch.save(model.module.state_dict(), best_model_path)
                    print('Best checkpoint saved')

                if epoch - best_epoch == early_stopping:
                    save_metrics(
                        losses, metrics_values, primary_metric_per_class,
                        losses_path, metrics_path, dataframe_path
                    )
                    print(f'Training is complete, best epoch: {best_epoch}, best {primary_metric}: {best_metric}')
                    return None

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(total_loss)
                    else:
                        scheduler.step()
                        
                    # Log learning rate to ClearML
                    if task is not None:
                        current_lr = optimizer.param_groups[0]['lr']
                        task.get_logger().report_scalar(
                            title="Learning Rate",
                            series="value",
                            value=current_lr,
                            iteration=epoch
                        )

                print()
    
    save_metrics(
        losses, metrics_values, primary_metric_per_class,
        losses_path, metrics_path, dataframe_path
    )
    print(f'Training is complete, best epoch: {best_epoch}, best {primary_metric}: {best_metric}')
    return None
