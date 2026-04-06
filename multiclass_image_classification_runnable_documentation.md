# Research Documentation for `multiclass_image_classification_runnable.ipynb`

## Abstract
This document provides a research-style technical interpretation of the notebook [`multiclass_image_classification_runnable.ipynb`](multiclass_image_classification_runnable.ipynb). The notebook implements an end-to-end convolutional neural network pipeline for multi-class image classification on small `16 x 16` RGB images. Its central design philosophy is to preserve `ReLU` as the activation function throughout the model while improving generalization through careful preprocessing, residual learning, modern optimization, and disciplined validation.  

The pipeline combines:
- reproducible training controls,
- stratified data splitting,
- train-split normalization,
- conservative augmentation suited to low-resolution imagery,
- a ReLU-based residual convolutional architecture,
- `AdamW`,
- `OneCycleLR`,
- `mixup`,
- mixed precision,
- gradient clipping,
- exponential moving average (EMA) of parameters,
- and optional horizontal-flip test-time augmentation (TTA).

The notebook is designed not merely to train a classifier, but to do so in a way that is experimentally defensible, reproducible, and explainable.

## 1. Research Context and Objective
The goal of the notebook is to solve a supervised multi-class image classification problem in which:
- the inputs are RGB images,
- the images are very small (`16 x 16`),
- the output is one label among multiple discrete classes,
- and the evaluation objective is strong validation accuracy with good generalization to a held-out test set.

Low-resolution image classification creates a particular methodological challenge: every pixel carries comparatively more information than in large-image problems, so aggressive geometric transformations or early downsampling may destroy class evidence rather than enhance robustness. The notebook therefore adopts a design that is explicitly tailored to small images.

The notebook’s broader scientific objective is not only to produce a high-performing model, but to do so using a transparent and auditable series of steps. Each component is therefore chosen for a reason:
- to reduce bias in evaluation,
- to stabilize training,
- to increase representational capacity without violating the `ReLU` requirement,
- and to improve validation performance without relying on opaque heuristics.

## 2. Computational Environment
This notebook was executed with an **NVIDIA A100 GPU**.

This is an important methodological detail because the A100 materially affects:
- the feasibility of longer training schedules,
- the practical use of mixed precision (`autocast` + `GradScaler`),
- the ability to sustain larger batch throughput,
- and the speed with which repeated validation and TTA-based inference can be performed.

The A100 does **not** change the mathematical logic of the model, but it substantially improves the runtime efficiency of the training loop. In a research setting, this matters because it enables broader hyperparameter exploration and more stable experimental practice.

## 3. End-to-End Pipeline Summary
At a high level, the notebook follows the sequence below:

1. Import all scientific computing, deep learning, and visualization dependencies.
2. Set deterministic seeds for reproducibility.
3. Resolve the data directory robustly across environments.
4. Load training and test arrays from serialized `.pkl` files.
5. Convert image tensors to the `NCHW` layout required by PyTorch.
6. Define augmentation and dataset logic specialized for tiny images.
7. Create a stratified validation split and estimate normalization statistics from the training split only.
8. Inspect a mini-batch visually as a sanity check.
9. Define a ReLU residual CNN with progressively wider stages.
10. Move the model to the selected device and summarize its size.
11. Configure the objective function, optimizer, scheduler, and regularization settings.
12. Train with `mixup`, EMA, mixed precision, gradient clipping, and early stopping.
13. Plot diagnostics to interpret optimization behavior.
14. Evaluate the model on the held-out validation set.
15. Evaluate the same trained model on the full training set without augmentation.
16. Generate test predictions and export a submission file.
17. Save the learned model parameters for later reuse.

This flow reflects a mature machine learning workflow: data handling, experimental control, model construction, optimization, evaluation, and artifact persistence are all treated as first-class concerns.

## 4. Algorithmic Foundations
This section defines the major algorithms and modeling ideas used in the notebook, explains what each one is, and states why it was included.

### 4.1 Stratified Train-Validation Split
**Definition**  
A stratified split partitions data such that class proportions are preserved across the training and validation subsets.

**Why it is used here**  
In multi-class problems, an unstratified split can accidentally over-represent or under-represent some classes in the validation set. That would distort the measured validation accuracy and make it harder to trust the result. Stratification makes the validation score a more faithful estimate of future generalization.

**Purpose in the model-building pipeline**  
It protects evaluation quality. The model may train identically either way, but the interpretation of its performance becomes more scientifically valid with stratification.

### 4.2 Channel-Wise Normalization
**Definition**  
Normalization rescales each input channel so that it has a standardized mean and standard deviation, usually based on the training distribution.

**Why it is used here**  
Neural networks optimize more reliably when inputs have a stable numeric scale. Using statistics from the training split only avoids leakage from validation into preprocessing.

**Purpose in the model-building pipeline**  
It improves optimization stability, helps BatchNorm work in a better input regime, and reduces sensitivity to raw brightness scale.

### 4.3 Conservative Data Augmentation
**Definition**  
Data augmentation produces modified versions of existing samples so that the network learns invariances rather than memorizing the exact training pixels.

**Augmentations used here**
- horizontal flip,
- reflection-padded random crop,
- mild brightness/contrast jitter.

**Why these choices are used here**  
For `16 x 16` images, strong transforms can destroy class identity. The notebook therefore uses limited, task-compatible perturbations rather than aggressive corruption.

**Purpose in the model-building pipeline**  
Augmentation improves generalization by teaching the model to be robust to small translations and modest illumination changes while preserving the semantics of the class.

### 4.4 Residual Learning
**Definition**  
Residual learning introduces shortcut connections so that a block learns a residual correction relative to its input rather than a completely new representation from scratch.

**Why it is used here**  
Residual connections make deeper CNNs easier to optimize by improving gradient flow and reducing the degradation problem.

**Purpose in the model-building pipeline**  
Residual blocks allow the notebook to use a stronger architecture than a shallow CNN while retaining stable training under `ReLU`.

### 4.5 ReLU Activation
**Definition**  
The Rectified Linear Unit is defined as `ReLU(x) = max(0, x)`.

**Why it is used here**  
The notebook explicitly preserves `ReLU` throughout the model as a design requirement. ReLU is simple, computationally efficient, and highly compatible with He/Kaiming initialization.

**Purpose in the model-building pipeline**  
It introduces nonlinearity so the model can learn complex decision boundaries while maintaining optimization simplicity.

### 4.6 Batch Normalization
**Definition**  
BatchNorm normalizes intermediate activations using batch statistics during training and running estimates during evaluation.

**Why it is used here**  
It stabilizes gradient propagation, reduces internal covariate shift, and allows moderately deeper architectures to train more efficiently.

**Purpose in the model-building pipeline**  
It improves convergence speed and reliability, especially in conjunction with ReLU and AdamW.

### 4.7 Dropout and Dropout2d
**Definition**  
Dropout randomly zeros activations during training to reduce co-adaptation. `Dropout2d` applies this idea at the feature-map level.

**Why it is used here**  
The model is wide enough to risk overfitting. Dropout provides explicit regularization in both convolutional and fully connected stages.

**Purpose in the model-building pipeline**  
It improves robustness and reduces memorization.

### 4.8 Global Average Pooling
**Definition**  
Global average pooling collapses spatial dimensions by averaging each feature map into one scalar.

**Why it is used here**  
It reduces the number of classifier parameters and encourages the model to learn globally meaningful channels rather than overfitting to fixed positions.

**Purpose in the model-building pipeline**  
It connects the convolutional backbone to a compact classifier head while controlling parameter growth.

### 4.9 Cross-Entropy Loss with Label Smoothing
**Definition**  
Cross-entropy compares predicted class probabilities with the target label distribution. Label smoothing softens the one-hot target slightly so the model is discouraged from becoming overconfident.

**Why it is used here**  
Overconfident classifiers often generalize worse. Label smoothing provides a mild regularization effect at the objective level.

**Purpose in the model-building pipeline**  
It improves calibration and can improve validation accuracy by discouraging brittle decision surfaces.

### 4.10 AdamW
**Definition**  
AdamW is an adaptive optimizer that decouples weight decay from the gradient update.

**Why it is used here**  
It is a strong practical optimizer for modern vision models, especially when paired with well-tuned learning-rate scheduling.

**Purpose in the model-building pipeline**  
It provides stable optimization while allowing effective regularization through explicit weight decay.

### 4.11 OneCycle Learning Rate Policy
**Definition**  
`OneCycleLR` changes the learning rate dynamically over training, typically warming up to a peak and then annealing down to a small value.

**Why it is used here**  
This schedule often improves convergence quality and allows efficient use of a moderately aggressive peak learning rate.

**Purpose in the model-building pipeline**  
It helps the optimizer explore parameter space early, then settle into a good minimum later.

### 4.12 Mixup
**Definition**  
Mixup creates virtual training samples by convexly interpolating pairs of inputs and their corresponding labels.

**Why it is used here**  
It regularizes the classifier without needing stronger image corruption. For tiny images, that is especially attractive because excessive augmentation can destroy information.

**Purpose in the model-building pipeline**  
It smooths the decision boundary and reduces overfitting.

### 4.13 Mixed Precision Training
**Definition**  
Mixed precision uses lower-precision arithmetic where safe, while preserving full precision where necessary for numerical stability.

**Why it is used here**  
On A100 hardware, mixed precision substantially improves throughput.

**Purpose in the model-building pipeline**  
It accelerates training and reduces memory pressure without changing the overall algorithmic design.

### 4.14 Gradient Clipping
**Definition**  
Gradient clipping limits the norm of gradients before the parameter update.

**Why it is used here**  
Aggressive schedules and mixed precision can occasionally produce unstable updates. Clipping helps prevent outlier steps.

**Purpose in the model-building pipeline**  
It improves training stability.

### 4.15 Exponential Moving Average (EMA)
**Definition**  
EMA maintains a smoothed copy of model parameters by updating them as a weighted moving average of recent parameter values.

**Why it is used here**  
Validation often improves when using the EMA model rather than the instantaneous latest weights, because EMA suppresses noisy parameter fluctuations.

**Purpose in the model-building pipeline**  
It serves as a more stable evaluation checkpoint and often improves generalization.

### 4.16 Test-Time Augmentation (TTA)
**Definition**  
TTA applies small, valid transformations at inference time and averages the predictions.

**Why it is used here**  
Since horizontal flips are already used during training, averaging original and flipped predictions is a low-risk way to reduce prediction variance.

**Purpose in the model-building pipeline**  
It slightly improves inference robustness.

### 4.17 Early Stopping
**Definition**  
Early stopping terminates training when validation performance stops improving for a predefined number of epochs.

**Why it is used here**  
It avoids wasting compute once the model has plateaued and helps prevent late-stage overfitting.

**Purpose in the model-building pipeline**  
It keeps the training run efficient and anchored to validation performance.

## 5. Cell-by-Cell Technical Walkthrough
This section explains what every notebook cell is doing, why it is needed, and which algorithms or methodological ideas it contributes.

### Cell 0: Notebook framing and design statement
This markdown cell states the problem and the notebook’s design philosophy. It identifies the central ingredients of the pipeline:
- `ReLU` activations,
- stratified validation,
- normalization,
- conservative augmentation,
- a residual CNN,
- and a stronger optimization loop.

Its purpose is conceptual rather than computational: it declares the methodological assumptions guiding the rest of the notebook.

### Cell 1: Imports, reproducibility, and device selection
This code cell imports all required libraries:
- Python utilities: `copy`, `os`, `pickle`, `random`, `Path`,
- scientific stack: `numpy`, `pandas`, `matplotlib`,
- PyTorch modules,
- `train_test_split` from scikit-learn.

It then defines a random seed and applies it to:
- Python random state,
- NumPy random state,
- PyTorch CPU state,
- PyTorch CUDA state.

Finally, it identifies whether training will run on CPU or GPU.

**Algorithms and concepts used**
- reproducibility controls,
- deterministic backend behavior,
- hardware-aware device assignment.

**Purpose**  
This cell establishes experimental discipline. Without it, later comparisons between runs would be much harder to interpret.

### Cell 2: Data-loading rationale
This markdown cell explains why data existence is checked before training begins. It improves usability and helps prevent obscure downstream failures.

**Purpose**  
It documents a defensive-programming principle: fail early when required inputs are absent.

### Cell 3: Portable path resolution
This code cell defines `resolve_base_path()`, which searches several likely directories for the challenge files and falls back to mounting Google Drive if running in Colab.

**Algorithms and concepts used**
- environment portability,
- runtime path discovery,
- explicit failure messaging.

**Purpose**  
This prevents the notebook from being tied to one machine layout and makes it portable across local and cloud environments.

### Cell 4: Loading serialized data
This code cell reads:
- `train_X_y.pkl`,
- `test_X.pkl`.

It then prints the raw shapes.

**Algorithms and concepts used**
- serialized array loading via Python pickle.

**Purpose**  
It transforms stored dataset artifacts into in-memory arrays that subsequent preprocessing can manipulate.

### Cell 5: Tensor layout conversion and label formatting
PyTorch convolutional layers expect image tensors in `NCHW` form:
- `N`: batch size,
- `C`: channels,
- `H`: height,
- `W`: width.

This cell:
- validates that each image array is 4D,
- converts NHWC to NCHW if needed,
- flattens labels,
- infers image size and number of classes,
- reports pixel range and shape information.

**Algorithms and concepts used**
- tensor layout transformation,
- shape validation,
- dtype conversion.

**Purpose**  
This cell ensures compatibility between raw data storage format and PyTorch’s convolutional computation model.

### Cell 6: Augmentation and dataset abstraction
This is one of the core preprocessing cells. It defines:
- `reflect_pad_crop()`,
- `random_color_jitter()`,
- `ImageDataset`.

#### `reflect_pad_crop()`
This function adds reflection padding and then randomly crops back to the original image size.

**Definition**  
Reflection padding extends the border by mirroring nearby values rather than filling with zeros.

**Purpose**  
It simulates small translations without introducing sharp artificial borders.

#### `random_color_jitter()`
This function performs mild brightness and contrast perturbation.

**Purpose**  
It teaches the model to be less sensitive to illumination differences while respecting the fragile information budget of tiny images.

#### `ImageDataset`
This custom PyTorch dataset:
- reads a sample,
- scales pixels to `[0, 1]`,
- applies optional augmentation,
- normalizes using the training statistics,
- returns an image tensor and label.

**Purpose**  
It unifies preprocessing, augmentation, normalization, and label handling into one reusable abstraction.

### Cell 7: Stratified split, train statistics, and DataLoaders
This cell operationalizes the experimental design.

It performs:
- a stratified train-validation split,
- channel-statistics estimation from the training split only,
- construction of train/validation/test datasets,
- construction of `DataLoader` objects.

It also sets:
- `VAL_SIZE = 0.12`,
- `BATCH_SIZE = 96`,
- `drop_last=True` for the training loader.

**Algorithms and concepts used**
- stratified sampling,
- leakage-free normalization,
- minibatch loading,
- GPU-friendly data transfer with `pin_memory`.

**Purpose**  
This cell creates the actual data pipeline used by the model.

### Cell 8: Visual sanity-check rationale
This markdown cell explains why inspecting a mini-batch matters.

**Purpose**  
It reflects a strong experimental habit: visualize the data after augmentation and normalization before trusting the optimization process.

### Cell 9: Batch visualization
This cell defines `imshow()` to reverse normalization for display and then visualizes a mini-batch from the training loader.

**Algorithms and concepts used**
- inverse normalization for interpretability,
- batch-grid visualization.

**Purpose**  
It confirms that the input pipeline is functioning sensibly.

### Cell 10: Model-design rationale
This markdown cell explains the architectural philosophy of the CNN:
- preserve spatial detail longer,
- use a wider residual model,
- move regularization into optimization,
- support TTA at inference.

**Purpose**  
This is the conceptual bridge between preprocessing and architecture.

### Cell 11: Residual block and full model definition
This is the main architecture cell.

#### `ResidualBlock`
This block contains:
- convolution,
- BatchNorm,
- ReLU,
- second convolution,
- BatchNorm,
- optional shortcut projection,
- dropout,
- residual addition,
- final ReLU.

**Purpose**  
It defines the repeated local computation unit of the CNN.

#### `WideSmallResNet`
This class defines the full model:
- a stem layer,
- four residual stages,
- global average pooling,
- a fully connected classifier head.

**Architectural logic**
- `layer1` preserves full spatial resolution,
- `layer2` downsamples to `8 x 8`,
- `layer3` downsamples to `4 x 4`,
- `layer4` increases representational richness without further shrinking.

**Purpose**  
This model balances capacity, regularization, and spatial preservation for tiny images.

### Cell 12: Device placement
This cell moves the network to the selected device.

**Purpose**  
It ensures the model and input tensors live on the same hardware device during training and inference.

### Cell 13: Model summary
This cell attempts to use `torchinfo.summary` and falls back to manual parameter counts if `torchinfo` is unavailable.

**Algorithms and concepts used**
- structural introspection,
- parameter counting.

**Purpose**  
It quantifies model scale and makes the architecture easier to inspect.

### Cell 14: Optimization rationale
This markdown cell explains why the optimization stack is more advanced than a basic supervised training loop.

It highlights:
- longer convergence time,
- `mixup`,
- EMA,
- TTA.

**Purpose**  
This cell documents the reasoning behind the optimizer and regularization design.

### Cell 15: Loss, optimizer, scheduler, and training hyperparameters
This cell defines the most important hyperparameters in the notebook:
- target accuracy,
- total epochs,
- max learning rate,
- weight decay,
- label smoothing,
- patience,
- gradient clipping threshold,
- mixup settings,
- EMA decay,
- TTA flag.

It then constructs:
- `CrossEntropyLoss`,
- `AdamW`,
- `OneCycleLR`,
- `GradScaler`.

**Purpose**  
This cell formalizes the training objective and optimization policy.

### Cell 16: Training-loop rationale
This markdown cell explains the higher-level design of the training loop:
- `mixup` for smoother decision boundaries,
- EMA for stable evaluation,
- batch-wise scheduler updates,
- checkpoint restore,
- TTA-enabled validation.

**Purpose**  
It explains not just what the code does, but why the loop is built that way.

### Cell 17: Training mechanics
This is the operational core of the notebook.

It defines:
- `ModelEMA`,
- `mixup_batch()`,
- `forward_with_tta()`,
- `run_train_epoch()`,
- `evaluate()`.

It then runs the full training loop and records:
- training loss,
- training accuracy,
- validation loss,
- validation accuracy,
- learning-rate history.

It also:
- tracks the best validation checkpoint,
- triggers early stopping,
- restores the best checkpoint,
- prints whether the target was reached.

**Purpose**  
This cell turns all prior design decisions into an executable optimization process.

### Cell 18: Diagnostic plotting
This cell produces four plots:
- train vs validation loss,
- train vs validation accuracy,
- learning rate trajectory,
- generalization gap.

**Purpose**  
The plots allow a researcher to reason about:
- underfitting,
- overfitting,
- schedule behavior,
- and whether the optimizer was still improving at the point of stopping.

### Cell 19: Validation-stage rationale
This markdown cell explains why the held-out evaluation step matters after checkpoint restoration.

**Purpose**  
It distinguishes model selection from final model assessment.

### Cell 20: Validation evaluation and per-class analysis
This cell:
- re-evaluates the restored best model,
- reports validation loss,
- reports overall validation accuracy,
- reports per-class accuracy.

**Purpose**  
This is the principal performance-assessment cell for model selection.

### Cell 21: Training-set evaluation rationale
This markdown cell explains why the model is also evaluated on the full training set without augmentation.

**Purpose**  
It provides a diagnostic check for overfitting via the train-validation gap.

### Cell 22: Full training-set evaluation
This cell evaluates the trained model on all training samples without augmentation and compares the result to validation performance.

**Purpose**  
It quantifies the extent to which the model may be memorizing the training data versus generalizing.

### Cell 23: Submission-generation rationale
This markdown cell explains why inference happens only after the best checkpoint is restored.

**Purpose**  
It ensures the exported predictions reflect the best validated model rather than the final raw training state.

### Cell 24: Test-time inference
This cell defines `predict_loader()` and runs inference over the unlabeled test set.

It uses:
- evaluation mode,
- no-gradient inference,
- optional TTA,
- argmax class prediction.

**Purpose**  
It converts trained model logits into discrete test labels.

### Cell 25: Submission export
This cell packages predictions into a DataFrame with:
- `rowId`,
- `label`.

It then writes the result to:
- `my_cnn_results_improved_v2.csv`.

**Purpose**  
It creates the final submission artifact for external evaluation.

### Cell 26: Model-saving rationale
This markdown cell explains why `state_dict()` is the preferred persistence format.

**Purpose**  
It documents the model serialization strategy in a reproducible and PyTorch-standard way.

### Cell 27: Model persistence
This cell saves the model parameters to:
- `my_cnn_model_improved_v2.pth`.

**Purpose**  
It preserves the trained model for later analysis, reloading, or deployment.

### Cell 28: Reload template
This cell provides commented code for reconstructing and loading the model later.

**Purpose**  
It closes the experimental loop by making the saved artifact reusable.

## 6. Scientific Interpretation of the Notebook Design
From a senior-research perspective, the notebook is not merely a code artifact; it is an experimental pipeline encoding several important hypotheses:

1. Small-image classification benefits from preserving early spatial detail.
2. Generalization improves when regularization is shifted away from destructive pixel corruption and toward objective-level and optimization-level techniques.
3. Residual architectures remain effective even when the input resolution is very small.
4. Validation quality depends as much on correct experimental design as on raw model power.
5. Modern training stability tools such as EMA, mixed precision, gradient clipping, and dynamic scheduling can materially improve practical performance.

These hypotheses are reflected not only in the architecture but in the entire notebook structure.

## 7. Strengths of the Notebook
The notebook exhibits several methodological strengths:
- it is reproducible,
- it uses leakage-aware normalization,
- it adopts stratified evaluation,
- it uses architecture and optimizer choices that are coherent with the image scale,
- it includes diagnostic visualization,
- it documents evaluation artifacts,
- and it saves the model in a reusable format.

These are all markers of good research and engineering practice.

## 8. Practical Limitations and Considerations
Despite its strengths, the notebook has limitations that are important to acknowledge.

### 8.1 Validation split dependence
The reported validation result depends on one specific split. A different stratified split might yield a slightly different score.

### 8.2 TTA assumption
Horizontal-flip TTA assumes that flips are semantically valid for the underlying classes. This is often true in natural-image problems, but not universally.

### 8.3 Resource dependence
The use of an A100 makes the notebook operationally efficient. Running the same notebook on weaker hardware would preserve the same logic, but training time would increase.

### 8.4 Mixed precision sensitivity
Mixed precision is generally stable on A100 hardware, but it introduces a layer of numerical complexity that should always be validated empirically.

## 9. Concluding Assessment
`multiclass_image_classification_runnable.ipynb` is a strong example of a modern, carefully structured image-classification notebook. It combines sound data handling, an appropriate small-image residual architecture, and a mature optimization strategy into a coherent experimental system.  

Its most important contribution is not any one trick in isolation. Rather, it is the integration of:
- principled preprocessing,
- a well-chosen ReLU residual CNN,
- robust validation practice,
- and advanced but well-motivated optimization mechanisms.

In that sense, the notebook reads less like an ad hoc trial-and-error script and more like a compact applied-research workflow. That is precisely why it is suitable for documentation in the style of a senior researcher.

## 10. Recommended File Association
This document is the companion research note for:
- [`multiclass_image_classification_runnable.ipynb`](multiclass_image_classification_runnable.ipynb)

Recommended companion artifacts:
- single-model predictions: `my_cnn_results_improved_v2.csv`
- single-model weights: `my_cnn_model_improved_v2.pth`
