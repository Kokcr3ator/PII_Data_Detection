# PII_Data_Detection with Transformers
This repository is inspired by a Kaggle competition hosted by Vanderbilt University and The Learning Agency Lab, which seeks automated solutions for PII detection in a dataset of about 22,000 student essays (<a href="https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data" target="_blank">Link</a>).

## Optimization Strategy

In all models, predictions are made by maximizing the probability score for each token. Specifically, for token $t$ in document $d$, the predicted class $y_{t,d}$ is determined as:


$y_{t,d} = \arg\max_{i = 1,...,13} v_{t,d}(i)$

where $v_{t,d}$ represents the vector of predicted probabilities, with $v_{t,d}(i) = \mathbb{P}(y_{t,d} = i)$.

## Posterior Optimization for $F_5$ Score

The metric used to evaluate the models is the $F_{\beta}$ Score with $\beta = 5$:
```math
\begin{equation}
F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{(\beta^2 \cdot precision) + recall}, \quad 0 \leq F_{\beta} \leq 1
\end{equation}
```

Although models are not trained explicitly to maximize the $F_5$ score, a posteriori optimization is employed. Given the higher importance of recall over precision in the $F_5$ score, especially considering the uneven misclassification costs, a strategy is devised to bias predictions towards PII classification.

## Prediction Strategy

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

This practice results in enhanced performance across all models, as demonstrated in the following table

| Model                | Train Loss | Val Loss | Recall  | Precision | F5val   | F5test | F5test+threshold |
|----------------------|------------|----------|---------|-----------|---------|--------|------------------|
| BERTbase all data    | 0.000400   | 0.006504 | 0.864611| 0.601679  | 0.850319| 0.821  | 0.828            |
| BERTbase only PII    | 0.045200   | 0.006527 | 0.903485| 0.433719  | 0.867353| 0.875  | 0.878            |
| DistilBERT all data  | 0.000200   | 0.005594 | 0.909877| 0.798229  | 0.905009| 0.866  | 0.877            |
| DistilBERT only PII  | 0.000200   | 0.010796 | 0.934391| 0.545914  | 0.909498| 0.905  | 0.906            |

For detailed implementation, please refer to ```Training.ipynb```, ```Inference.ipynb``` and ```Optimal_threshold.ipynb```.



