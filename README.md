# AI-Image-Detector

## Repository Sturcture
```
AI-Image-Detector/
├── documents/                 # Project paper
├── eda/                       # Exploratory Data Analysis notebooks
├── images/                    # Figures
├── networks/                  # ResNet50, EfficientNetB3 networks
├── resources/                 # addtional resources
├── utils/                     # Utility classes
├── weights/                   # Pre-trained EfficientNetB3 encoder weight
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
├── SupCon.py                  # SupCon backbone training script
├── SupConClassifier.py        # Linear Classifier training script using SupCon features
├── losses.py                  # SupCon loss function definition
└── requirements.txt           # Python dependencies
```


## Objective

AI has reached a point where it can generate highly realistic faces, scenes, and objects. This study addresses the problem of distinguishing AI-generated visuals from authentic photographs using a unique dataset, "AI vs. Human-Generated Images," from a Kaggle competition. Unlike conventional datasets, this dataset provides paired images where each real image has a corresponding AI-generated counterpart, allowing for direct comparative analysis. We leverage this structured pairing within a deep learning framework, incorporating convolutional neural networks (CNNs) and transformer-based architectures to develop robust classifiers. In addition, we explore contrastive learning to enhance feature discrimination, hypothesizing that it improves generalization by enforcing a more distinct separation between real and AI-generated images.

## Data Source

The [AI vs. Human-Generated Images](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset?select=test_data_v2) dataset is published on Kaggle.

"The dataset consists of authentic images sampled from the Shutterstock platform across various categories, including a balanced selection where one-third of the images feature humans. These authentic images are paired with their equivalents generated using state-of-the-art generative models. This structured pairing enables a direct comparison between real and AI-generated content, providing a robust foundation for developing and evaluating image authenticity detection systems."

![Sample Dataset](images/dataset.png)


## Data Analysis

Exploratory Data Analysis (EDA) can be found [here](https://github.com/jh000107/AI-Image-Detector/blob/master/eda_V2.ipynb)

## Data Processing
