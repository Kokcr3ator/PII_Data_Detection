# PII_Data_Detection with Transformers
This repository is inspired by a Kaggle competition hosted by Vanderbilt University and The Learning Agency Lab, which seeks automated solutions for PII detection in a dataset of about 22,000 student essays (<a href="https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data" target="_blank">Link</a>).

## Optimization Strategy

In all models, predictions are made by maximizing the probability score for each token. Specifically, for token $t$ in document $d$, the predicted class $y_{t,d}$ is determined as:


$y_{t,d} = \arg\max_{i = 1,...,13} v_{t,d}(i)$

where $v_{t,d}$ represents the vector of predicted probabilities, with $v_{t,d}(i) = \mathbb{P}(y_{t,d} = i)$.

## Posterior Optimization for $F_5$ Score

Although models are not trained explicitly to maximize the $F_5$ score, a posteriori optimization is employed. Given the higher importance of recall over precision in the $F_5$ score, especially considering the uneven misclassification costs, a strategy is devised to bias predictions towards PII classification.

## Non-Classical Prediction Strategy

During inference, scores for all labels are computed for each token. However, only probabilities for PII labels are considered when applying the argmax function. This approach, outlined as:


$z_{t,d} = \arg\max_{i = 1,...,12} v_{t,d}(i)$

facilitates improved predictions by prioritizing PII classifications.

## Threshold-based Prediction

A threshold $\tau$ is introduced as a hyperparameter, fine-tuned using the validation set. Final predictions are made based on whether the probability for the PII label exceeds this threshold:

```math
y_{t,d} = \begin{cases} 
    z_{t,d} & \text{if } v_{t,d}(13) < \tau \\
    13 & \text{otherwise} 
\end{cases}
```

This practice results in enhanced performance across all models, as demonstrated in the provided metrics.

For detailed implementation, please refer to ```Training.ipynb``` and ```Inference.ipynb```.



