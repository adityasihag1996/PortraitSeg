import torch

def evaluate_mean_iou(model, dataloader, device):
    model.eval()
    
    total_iou = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, masks = batch
            inputs, masks = inputs.to(device), masks.to(device)

            outputs = model(inputs)
            predicted_masks = (outputs > 0.5).float()

            batch_iou = calculate_iou(predicted_masks, masks)

            total_iou += batch_iou.item()
            total_samples += inputs.size(0)

            del inputs, masks, outputs, predicted_masks

    mean_iou = total_iou / total_samples

    return mean_iou

def calculate_iou(predicted, target):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) - intersection
    
    iou = (intersection + 1e-15) / (union + 1e-15)
    
    return iou