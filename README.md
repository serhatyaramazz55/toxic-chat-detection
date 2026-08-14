# Toxic Chat Detection

A Natural Language Processing project for binary **toxic / non-toxic text classification**, with a focus on game-chat style messages.

The project compares classical machine-learning approaches with transformer-based models and evaluates the final model on a held-out unseen test set.

## Project Pipeline

- Prepare and merge multiple toxicity / gaming-text datasets
- Remove missing values and duplicate text samples
- Build stratified train, validation, and unseen test splits
- Train TF-IDF + Logistic Regression baseline
- Train TF-IDF + Linear SVM baseline
- Perform ROC, Precision-Recall, learning-curve, and overfitting analyses
- Tune classical models with `GridSearchCV`
- Fine-tune `distilroberta-base`
- Fine-tune `roberta-base` with early stopping
- Evaluate the selected RoBERTa model on the unseen test set
- Provide an interactive Gradio demo

## Dataset Summary

After merging and cleaning the data, the notebook produced **62,880** unique labeled text samples:

- Non-toxic: **34,799**
- Toxic: **28,081**

The resulting split was:

| Split | Samples |
|---|---:|
| Train | 47,788 |
| Validation | 11,948 |
| Unseen test | 3,144 |

## Model Results

### Validation Results

| Model | Accuracy | F1 |
|---|---:|---:|
| Logistic Regression (base) | 0.9057 | 0.8894 |
| Linear SVM (base) | 0.9190 | 0.9064 |
| DistilRoBERTa (best checkpoint by eval loss) | 0.9302 | 0.9230 |
| RoBERTa (best checkpoint) | **0.9448** | **0.9377** |

For RoBERTa, the best checkpoint occurred around epoch 2 with validation loss `0.2025`.

### Unseen Test Results — RoBERTa

| Metric | Score |
|---|---:|
| Accuracy | **0.9377** |
| Precision | **0.9467** |
| Recall | **0.9117** |
| F1 | **0.9289** |

## Technologies

- Python
- Pandas / NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Matplotlib
- Gradio

## Repository Structure

```text
toxic-chat-detection/
├── notebooks/
│   └── toxic_chat_detection.ipynb
├── data/
│   └── README.md
├── results/
├── .gitignore
├── README.md
└── requirements.txt
```

## Running the Notebook

The notebook was developed in **Google Colab**.

1. Open `notebooks/toxic_chat_detection.ipynb` in Colab.
2. Upload the dataset files described in `data/README.md`.
3. Install the dependencies if they are not already available.
4. Run the notebook cells in order.

If you run the notebook outside Colab, replace `/content/...` paths with paths appropriate for your environment.

## Notes

Large datasets, trained model weights, checkpoints, and generated archives are intentionally excluded from Git version control.

The notebook contains the full experiment workflow, including classical baselines, transformer fine-tuning, model comparison, unseen-test evaluation, and the Gradio demo.
