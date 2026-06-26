import torch

from metrics import evaluate_metrics

preds = torch.randn(2, 2, 512, 512)

labels = torch.randint(
    0,
    2,
    (2, 512, 512)
)

metrics = evaluate_metrics(preds, labels)

print(metrics)