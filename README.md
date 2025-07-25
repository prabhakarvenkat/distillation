
# Knowledge Distillation for CIFAR-10 Classification

This repository contains code for training and evaluating convolutional neural networks on the CIFAR-10 dataset, with a focus on knowledge distillation techniques. We explore different distillation methods, including:

-   **Standard Training:** Training a "Deep" (Teacher) and "Light" (Student) network independently using Cross-Entropy loss.
-   **Cosine Embedding Loss:** Distilling knowledge from the Teacher to the Student by minimizing the cosine distance between their hidden representations.
-   **Mean Squared Error (MSE) Loss for Feature Maps:** Distilling knowledge by regressing the Student's feature maps to match the Teacher's feature maps.

## Table of Contents

-   [Installation](#installation)
-   [Dataset](#dataset)
-   [Models](#models)
-   [Training](#training)
-   [Evaluation](#evaluation)
-   [Inference on Custom Images](#inference-on-custom-images)
-   [ONNX Export and Quantization](#onnx-export-and-quantization)
-   [Results](#results)

## Installation

To set up the environment, run the following command:

```bash
%pip install onnx torch torchvision tqdm
````

## Dataset

The CIFAR-10 dataset is used for training and evaluation. It will be automatically downloaded to the `./data` directory.

The dataset is preprocessed with standard normalization using mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

Additionally, the code includes a section to calculate custom normalization metrics for the MNIST dataset (though this part is not directly used for CIFAR-10 training).

## Models

Two main CNN architectures are defined:

  - **`DeepNN` (Teacher Network):** A larger convolutional neural network with more layers and parameters.
  - **`LightNN` (Student Network):** A smaller, more lightweight convolutional neural network.

Modified versions of these networks (`ModifiedDeepNN`, `ModifiedLightNN`, `ModifiedDeepNNRegressor`, `ModifiedLightNNRegressor`) are introduced for knowledge distillation experiments. These modified versions include mechanisms to extract intermediate hidden representations or feature maps.

## Training

The training process involves:

1.  **Independent Training:** Both `DeepNN` and `LightNN` are trained independently using `nn.CrossEntropyLoss`.
2.  **Knowledge Distillation with Cosine Loss:** The `ModifiedLightNN` (student) is trained to mimic the `ModifiedDeepNN` (teacher) using a combination of `nn.CrossEntropyLoss` for labels and `nn.CosineEmbeddingLoss` for hidden representations.
3.  **Knowledge Distillation with MSE Loss (Feature Map Regression):** The `ModifiedLightNNRegressor` (student) is trained to mimic the `ModifiedDeepNNRegressor` (teacher) using a combination of `nn.CrossEntropyLoss` for labels and `nn.MSELoss` for feature maps.

The `train` function handles standard training, while `train_cosine_loss` and `train_mse_loss` implement the respective knowledge distillation methods.

**Hyperparameters:**

  - `epochs`: 50
  - `learning_rate`: 5e-5
  - `batch_size`: 128
  - `hidden_rep_loss_weight` (for cosine loss): 0.25
  - `ce_loss_weight` (for cosine and MSE loss): 0.75
  - `feature_map_weight` (for MSE loss): 0.25

## Evaluation

The `test` function evaluates the model's accuracy on the test set. The `test_outputs` function is a similar evaluation function specifically for models that return multiple outputs (logits and hidden representations/feature maps).

## Inference on Custom Images

The `predict_image` function allows for inference on individual images. It loads a pre-trained model, applies the necessary transformations, and returns the predicted class and probabilities.

**Usage:**

```python
result = []
model_path = "./NNDeep.pth" # or "./NNLight.pth", etc.
folder_path = "./EvalDataset" # Directory containing images for prediction

for filename in os.listdir(folder_path):
    if filename.endswith((".png",".jpg",".jpeg")):
        image_path = os.path.join(folder_path,filename)
        predicted_label, probabilities = predict_image(image_path, model_path)
        result.append({
            "image" : filename,
            "predicted_class":predicted_label,
            "probabilities" : probabilities.tolist()
        })
print(result)
```

**Note:** For running this section, you will need to create an `EvalDataset` directory and place some image files in it.

## ONNX Export and Quantization

The `NNDeep.pth` model can be exported to ONNX format and dynamically quantized for optimized inference.

  - `ImagePredictor.onnx`: The ONNX model without quantization.
  - `ImagePredictorQuantizedFinalized.onnx`: The dynamically quantized ONNX model.

## Results

The following accuracies are reported:

  - Teacher accuracy: 71.77%
  - Student accuracy without teacher: 66.32%
  - Student accuracy with CE + CosineLoss: 65.00%
  - Student accuracy with CE + RegressorMSE: 65.98%

These results indicate the performance of the various training strategies.

## Saved Models

  - `final_BLSTM_model.pth`: The trained `DeepNN` model (initially saved with this name).
  - `NNDeep.pth`: The trained `DeepNN` model (saved again later).
  - `NNLight.pth`: The independently trained `LightNN` model.
  - `ModefiedNNLight.pth`: The `ModifiedLightNN` trained with CE + Cosine Loss.
  - `ModifiedNNLightReg.pth`: The `ModifiedLightNNRegressor` trained with CE + MSE Loss.
  - `ImagePredictor.onnx`: ONNX export of the `NNDeep` model.
  - `ImagePredictorQuantizedFinalized.onnx`: Quantized ONNX export of the `NNDeep` model.

<!-- end list -->
