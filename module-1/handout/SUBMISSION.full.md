# Module 1: Machine Learning, Full Assignment

**Grading notice:** The score shown on Gradescope is an AI-assisted reference score and is subject to change. The teaching assistants and instructor may review and adjust it and retain final authority over interpretation of the assignment requirements, grading rubric, and final grades. Correct, reasonable work can receive full marks; optional extras and writing polish are not required.

Submit exactly one ZIP containing the completed PyTorch implementation,
`report.pdf`, `report.tex`, every figure and LaTeX dependency used by the
report, and the required checkpoint files. Include the `models/`, `scripts/`,
and `model_ckpt/` structure described in the starter handout. Do not include
the CIFAR-10 dataset.

The ZIP must be self-contained for review: any `\includegraphics`, bibliography,
custom style, or data file needed to audit the report must be present. A single
outer directory named with your student ID is allowed.

The implementation must train and evaluate the classification model and record
the experiment with WandB. The report should document the architecture,
training process, WandB curves, and evaluation results.

This Gradescope assignment is worth 10 points. The autograder checks the
reproducible code and checkpoint requirements; DeepSeek grades the report's
curves and analysis using its text and rendered pages. TA/instructor review
can change the reference score.

## Required checkpoint and scoring

Include both files from the same training run:

```
model_ckpt/cifar10_SimpleCNN.pth
model_ckpt/cifar10_SimpleCNN_arg.pickle
```

Save the weights with `torch.save(model.state_dict(), path)` and the constructor
hyperparameter dictionary with `pickle.dump(model_hps, file)`. Create
`model_ckpt/` before saving. The grader loads these files on CPU; it does not
train your submitted model. Missing either file loses the entire 2-point
accuracy component, so otherwise complete work receives at most 8/10.

| Component | Points |
| --- | ---: |
| Data preparation, CNN implementation, training and evaluation scripts | 2.00 |
| CNN checkpoint accuracy | 2.00 |
| ViT implementation | 3.00 |
| Report: experiment description and WandB evidence (AI) | 1.00 |
| Report: CNN–ViT comparison and analysis (AI) | 2.00 |
| Total | 10.00 |

CNN accuracy is measured independently on all 10,000 official CIFAR-10 test
images with `ToTensor()` and `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`.
The model must have fewer than 100,000 parameters. Accuracy of at least 73%
earns 2/2; 67% earns 1.4/2; 60% or below earns 0/2, with linear
interpolation between these thresholds. Keep the required 75 training epochs.
The required report contributes 3 of the 10 points. DeepSeek awards full
report credit for correct, reasonable work; it does not require elaborate
writing or optional enhancements. Deductions identify substantive errors or
missing required evidence. Use the course LaTeX template and submit both
PDF and source; missing either receives zero report points until supplied.
Citations are not a separate automatic requirement. The optional composition
model is not required for 10/10.

If the AI service fails or cannot judge specific evidence reliably, the grade
is marked **pending review**, not a final zero for that component. A TA can
review it or rerun grading. PDF text and rendered report pages are sent to
DeepSeek for report grading. Reports exceeding the automatic processing
limits (20 rendered pages or 80,000 extracted characters) receive manual
review rather than an automatic deduction.

Module dropout (`nn.Dropout`) and equivalent functional dropout
(`torch.nn.functional.dropout` with `training=self.training`) are both accepted.
Dropout must be active during training and disabled during evaluation.

For `prepare_cifar10_datasets`, return 50,000 training images, 8,000 validation
images, and the remaining 2,000 test images, following the starter's 8:2 TODO.
The earlier implementation that returns the full 10,000-image test dataset as
the third result is also accepted. In either case, checkpoint scoring uses the
same full 10,000-image test dataset. A local accuracy measured only on the
2,000-image subset can therefore differ from the Gradescope accuracy.

## Optional Composition bonus

The optional SimpleComposition task is reviewed manually by the TA and can
earn up to **one additional course point**, as described in the full handout.
It is separate from the required 10-point assignment and is not awarded by
the autograder. If attempting it, include the completed `SimpleComposition.py`
and CNN `forward_map`, the matching
`model_ckpt/cifar10_SimpleComposition.pth` and
`model_ckpt/cifar10_SimpleComposition_arg.pickle`, and a clearly labeled bonus
section in your report with the training curves and test accuracy. The TA
checks the implementation, the 120K parameter constraint, and the published
bonus accuracy curve before recording the course bonus separately.
