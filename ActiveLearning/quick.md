# Quick papers

## [Active Labeling: Streaming Stochastic Gradients](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6fee03d84375a159ecd3769ebbacae83-Abstract-Conference.html)

Use the magnitude of a random projection of the model's gradients as a learning rate multiplier.

## [Active Learning for Multiple Target Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/faacb7a4827b4d51e201666b93ab5fa7-Abstract-Conference.html)

1. Shortlist datapoints which are most disagreed among a mixture of models
2. Shortlist models based on accuracy difference from the best model so far

## [Active Learning Helps Pretrained Models Learn the Intended Task](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b43a0e8a35b1c044b18cd843b9771915-Abstract-Conference.html)

active learning on untrained models do more harm than good

## [Efficient Active Learning with Abstention](https://proceedings.neurips.cc/paper_files/paper/2022/file/e5aa7171449b83f8b4eec1623eac9906-Supplemental-Conference.pdf)

If the model predicts "I dont know" then dont lookup the real groundtruth y.

## [Active Learning Through a Covering Lens](https://arxiv.org/abs/2205.11320)

1. Semantically embed all points
2. Pick the points that have most neighbours within a sphere

## [Active Surrogate Estimators: An Active Learning Approach to Label–Efficient Model Evaluation](https://arxiv.org/abs/2202.06881)

1. Have an auxilary model
2. Peturb the auxilary model's parameters
3. Measure disagreement between the two, scale the learning rate for that sample with this
4. Also, penalize the primary model more if the confidence of aux model is high.
