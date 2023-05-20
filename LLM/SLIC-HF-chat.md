# SLIC

### You
https://arxiv.org/pdf/2305.10425.pdf

### ChatGPT
I have successfully downloaded and stored the document. Now, if you have any specific questions or need information from this document, feel free to ask!

### You
how is it different from RLHF?

### ChatGPT
The document discusses the differences between RLHF (Reinforcement Learning from Human Feedback) and the proposed method, SLiC-HF (Sequence Likelihood Calibration from Human Feedback). Here are the key differences:

1. Training Complexity: RLHF-PPO introduces significant complexity to the training process by adding separate value and reward networks that may be comparable in size to the policy network. These networks are typically kept in memory to maximize training speed, which for a given memory budget significantly reduces the maximum size of the trainable model. The optimization steps are also significantly slower due to the use of roll-outs in the training loop. In contrast, SLiC-HF avoids this complexity and is simpler to implement and easier to tune (Page 2).
2. Performance: SLiC-HF-sample-rank with reward model has similar performance as RLHF-PPO and SLiC-HF-sample-rank with ranking model is better than the RLHF-PPO. The summaries from SLiC-HF models are slightly longer than the RLHF-PPO model, and their length controlled win rate is similar to RLHF-PPO (Page 7).
3. Compute and Memory Efficiency: SLiC-HF is more compute and memory efficient than RLHF-PPO. In RLHF-PPO, the policy, value, reward, and SFT models are used within the PPO training loop and are often held in hardware memory to ensure faster training steps. In contrast, in SLiC-HF, the rewards can be computed completely in parallel and offline, thus using 1/4 the memory for model weights during training. Such memory savings could be re-purposed to train larger models (Page 8).
4. Pairwise Ranking vs Reward model: RL algorithms seek to maximize the expected reward of trajectories, in this case, the human judgement in the quality of model summaries. This reward function typically is assumed to be pointwise, whereas human preference data is collected pairwise to improve reliability. Thus there is noise introduced in converting pairwise judgements into pointwise rewards. Since SLiC-HF only cares about the relative rank of two summaries, this pairwise-to-pointwise noise is avoided (Page 8).
5. Value of States and Actions in Language: SLiC-HF does not rely on a sub-model and only uses the cleaner preference signal to drive parameter updates, leading to what is conjectured to be more stable optimization (Page 8).

In conclusion, SLiC-HF presents a competitive alternative to the RLHF-PPO implementation while being simpler to implement, easier to tune, and computationally efficient (Page 9).

### You
How is SLIC-hF implemented? 

### ChatGPT
The implementation of SLiC-HF (Sequence Likelihood Calibration from Human Feedback) involves several steps:

1. Human Evaluation: Side-by-side human evaluation is conducted between multiple systems using crowd-sourcing. Given a document and 2-4 summaries, raters are tasked to assign a pointwise overall quality to each summary, select if the summary is factual or not, and choose the best summary. Each task is replicated and judged by 3 different raters. To eliminate bias, all the models are anonymized and the order of summaries for each task is randomly shuffled (Page 6).
2. SLiC-HF Ablation Study: A 4-way side-by-side human evaluation is conducted to confirm the ablation results. 100 examples from the validation set are sampled from reference, SFT model, continue fine-tuning model, and SLiC-HF model (SLiC-HF-sample-rank, using ranking model, regularized on best decodes). SLiC-HF is chosen as the best model 73% of the time, has significantly higher average quality, and is the most factual model (Page 6).
3. SLiC-HF vs RLHF-PPO: The performance of SLiC-HF is compared with the RLHF-PPO model. SLiC-HF-sample-rank with reward model has similar performance as RLHF-PPO and SLiC-HF-sample-rank with ranking model is better than the RLHF-PPO. The summaries from SLiC-HF models are slightly longer than the RLHF-PPO model, and their length controlled win rate is similar to RLHF-PPO (Page 7).
4. Scaling Up SLiC: Two ways of scaling up the SLiC-HF-sample-rank are studied: (1) scaling up generation model parameters, (2) scaling up the number of decoded candidates. Scaling up generation model from 770M to 11B significantly improves both the SFT model and the SLiC-HF model. On the other hand, scaling up the number of decoded candidates from 8 to 64 does not help much (Page 7).
5. Compute/Memory Efficiency and Parallelism: The compute and memory efficiency differences between SLiC-HF and RLHF-PPO are summarized. SLiC-HF is more compute and memory efficient than RLHF-PPO (Page 7).

In conclusion, SLiC-HF is implemented through a combination of human evaluation, ablation studies, performance comparison with RLHF-PPO, and scaling strategies. It is designed to be simpler to implement, easier to tune, and more computationally efficient than RLHF-PPO (Page 9).

### You
what is the loss function? 

### ChatGPT
The loss function for SLiC-HF (Sequence Likelihood Calibration from Human Feedback) is defined as follows:

L(θ) = max(0, δ − log Pθ(y+|x) + log Pθ(y−|x)) − λ log Pθ(yref|x)

Here's a breakdown of the terms:

- The first term is the calibration loss where x is the input sequence, y+ and y− are positive and negative sequences, and δ is a hyper-parameter for the margin of the ranking loss.
- The second term is the cross-entropy loss, where yref is some target sequence and λ is the regularization weight. Cross-entropy loss encourages the model to stay close to the SFT (Supervised Fine-Tuning) model, similar to the KL term used in Stiennon et al. [18], however it does not need an extra copy of SFT weights.

The choices of y+ and y− are discussed in subsections 2.2 and 2.3 of the document. The choices of regularization target yref is discussed in subsection 2.4 (Page 3).

In addition to this, the document also mentions the loss function for the reward model used by RLHF (Reinforcement Learning from Human Feedback):

loss(rφ) = −E(x,y+,y−)∼DHF [log(σ(rφ(x, y+) − rφ(x, y−))]

This is the expected log probability of the reward model assigning a higher score to the positive sequence y+ than the negative sequence y−, given the context x (Page 2).

### You
Could you implement python code for SLIC loss?

### ChatGPT
Sure, here is a simple implementation of the SLiC-HF loss function in Python using PyTorch:

```python
import torch
import torch.nn.functional as F

def slic_loss(model, x, y_pos, y_neg, y_ref, delta, lambda_reg):
    """
    SLiC-HF loss function.

    Args:
        model: The model being trained.
        x: The input sequence.
        y_pos: The positive sequence.
        y_neg: The negative sequence.
        y_ref: The target sequence.
        delta: The margin for the ranking loss.
        lambda_reg: The regularization weight.

    Returns:
        The computed loss.
    """
    # Calculate the log probabilities of the sequences
    log_prob_y_pos = model(x, y_pos).log_prob(y_pos)
    log_prob_y_neg = model(x, y_neg).log_prob(y_neg)
    log_prob_y_ref = model(x, y_ref).log_prob(y_ref)

    # Calculate the calibration loss
    calibration_loss = F.relu(delta - log_prob_y_pos + log_prob_y_neg)

    # Calculate the regularization loss
    regularization_loss = lambda_reg * log_prob_y_ref

    # Combine the losses
    loss = calibration_loss - regularization_loss

    return loss

```

This is a simplified example and assumes that the model has a `log_prob` method that calculates the log probability of a sequence given an input. In practice, you would need to adjust this to match the specifics of your model and data.

Remember that this is a pairwise loss function, so you need to provide a positive and a negative sequence for each input. The positive sequence should be one that is preferred over the negative sequence according to some criterion (e.g., human feedback). The target sequence `y_ref` is used for regularization to prevent the model from deviating too much from the initial policy. The `delta` and `lambda_reg` parameters control the margin for the ranking loss and the weight of the regularization term, respectively.

### You
what do the dimensions of log_prob_y_pos  tensor do?

### ChatGPT
The `log_prob_y_pos` tensor represents the log probabilities of the positive sequence `y_pos` given the input sequence `x` as calculated by the model.

The dimensions of this tensor would typically match the dimensions of your sequence `y_pos`. For instance, if `y_pos` is a sequence of length `n`, then `log_prob_y_pos` would also be a tensor of shape `(n,)`.

Each element in the `log_prob_y_pos` tensor corresponds to the log probability of a particular token in the `y_pos` sequence given the input `x` and the preceding tokens in the sequence.

This is under the assumption that the model is an autoregressive model, where each token is generated one at a time, conditioned on the input and all previously generated tokens. This is the case for many popular language models like GPT and Transformer models.

Please note that the actual dimensions might vary depending on the specific implementation of the `log_prob` method in your model and the structure of your data.

### You
could you do a torch implementation of model.log_prob function?

### ChatGPT
Sure, here's a simplified example of how you might implement a `log_prob` method in a PyTorch model. This example assumes that your model is a language model that outputs logits for each token in the vocabulary, and that your sequences are encoded as integer tensors where each integer represents a token.

```python
def log_prob(self, x, y):
    """
    Calculate the log probability of a sequence y given an input x.

    Args:
        x: The input sequence.
        y: The target sequence.

    Returns:
        The log probability of y given x.
    """
    # Forward pass through the model
    logits = self.forward(x)

    # Calculate the probabilities using a softmax function
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Gather the probabilities of the actual tokens in y
    y_probs = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)

    # Calculate the log probabilities
    log_probs = torch.log(y_probs)

    # Sum the log probabilities for each token in the sequence
    sequence_log_prob = log_probs.sum(dim=-1)

    return sequence_log_prob

```

This function calculates the log probability of each token in the sequence `y` given the input `x` and the preceding tokens, and then sums these log probabilities to get the log probability of the entire sequence.

Please note that this is a simplified example and might not work out-of-the-box depending on your specific model and data. For example, this function assumes that the sequences `x` and `y` are already properly encoded and aligned, and that the model's `forward` method returns logits for each token in the vocabulary. You might need to adjust this code to fit your specific needs.

### You
explain this line     y_probs = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)

### ChatGPT
Sure, let's break down this line of code:

```python
y_probs = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)

```

1. y.unsqueeze(-1): This operation adds an extra dimension to the tensor y at the last axis. If y is a tensor of shape (batch_size, sequence_length), after this operation, it will become a tensor of shape (batch_size, sequence_length, 1). This is done to match the shape requirements of the gather function.
2. probs.gather(-1, y.unsqueeze(-1)): The gather function is used to collect elements from probs tensor. The first argument -1 specifies the dimension to gather from. Here, it's the last dimension, which usually represents the vocabulary size in the output of a language model. The second argument is the indices to gather, which is y.unsqueeze(-1) in this case. Essentially, for each position in the sequence, this operation is picking out the probability of the actual token in y from the probs tensor.
3. .squeeze(-1): This operation removes the last dimension of size 1 from the tensor. If the tensor is of shape (batch_size, sequence_length, 1), after this operation, it will become a tensor of shape (batch_size, sequence_length). This is done to remove the extra dimension added by unsqueeze(-1) operation.

So, in summary, this line of code is picking out the probabilities of the actual tokens in the sequence `y` from the output probabilities of the model, and storing them in the tensor `y_probs`.

### You
Can you write the RLHF loss as torch?

### ChatGPT
Sure, here is a simple implementation of the RLHF (Reinforcement Learning from Human Feedback) loss function in Python using PyTorch:

```python
import torch
import torch.nn.functional as F

def rlhf_loss(reward_model, x, y_pos, y_neg):
    """
    RLHF loss function.

    Args:
        reward_model: The reward model.
        x: The input sequence.
        y_pos: The positive sequence.
        y_neg: The negative sequence.

    Returns:
        The computed loss.
    """
    # Calculate the rewards for the sequences
    reward_y_pos = reward_model(x, y_pos)
    reward_y_neg = reward_model(x, y_neg)

    # Calculate the difference in rewards
    reward_diff = reward_y_pos - reward_y_neg

    # Calculate the loss
    loss = -torch.log(F.sigmoid(reward_diff))

    return loss

```

This is a simplified example and assumes that the reward model has a method that calculates the reward of a sequence given an input. In practice, you would need to adjust this to match the specifics of your reward model and data.

Remember that this is a pairwise loss function, so you need to provide a positive and a negative sequence for each input. The positive sequence should be one that is preferred over the negative sequence according to some criterion (e.g., human feedback).

### You
    log_prob_y_pos = model(x, y_pos).log_prob(y_pos)
    log_prob_y_neg = model(x, y_neg).log_prob(y_neg)
    log_prob_y_ref = model(x, y_ref).log_prob(y_ref)

What is y_ref and what purpose does it serve?

### ChatGPT
In the context of the SLiC-HF (Sequence Likelihood Calibration from Human Feedback) loss function, `y_ref` is a reference or target sequence. It's used for regularization to prevent the model from deviating too much from the initial policy.

The regularization term in the loss function is `- λ log Pθ(yref|x)`, where `λ` is the regularization weight, `Pθ(yref|x)` is the probability of the reference sequence `y_ref` given the input `x` according to the current model parameters `θ`, and `log` is the natural logarithm. This term encourages the model to assign a high probability to the reference sequence, effectively pulling the model's predictions towards the reference.

The reference sequence could be, for example, the output of a pre-trained model or a human-generated sequence. The idea is that this sequence is known to be good according to some criterion, and we want our model to produce similar outputs. By including this term in the loss function, we're telling the model "don't stray too far from this reference".

In practice, the choice of the reference sequence and the regularization weight `λ` would depend on the specific task and data.

### You
In the paper what is used for y_pos, y_neg and y_ref? 

### ChatGPT
In the paper, the following are used for `y_pos`, `y_neg`, and `y_ref`:

1. y_pos and y_neg: These are positive and negative sequences determined from the human feedback data. The paper discusses two approaches for this:SLiC-HF with Sample and Rank: This approach samples candidates from the training split of the Supervised Fine-Tuning (SFT) model. The ranking of these candidates is determined by two text-to-text models trained from the human preference data. One model is a pointwise reward model that binarizes each ranked pair into a positive and a negative sequence. The other model is a pairwise ranking model that formulates the human feedback into a pairwise ranking problem.SLiC-HF Directly On Human Feedback: This approach directly calibrates on positive and negative sequences from the human feedback dataset, without a ranking or reward model.
2. SLiC-HF with Sample and Rank: This approach samples candidates from the training split of the Supervised Fine-Tuning (SFT) model. The ranking of these candidates is determined by two text-to-text models trained from the human preference data. One model is a pointwise reward model that binarizes each ranked pair into a positive and a negative sequence. The other model is a pairwise ranking model that formulates the human feedback into a pairwise ranking problem.
3. SLiC-HF Directly On Human Feedback: This approach directly calibrates on positive and negative sequences from the human feedback dataset, without a ranking or reward model.
4. y_ref: This is the target sequence for regularization. The paper considers two choices for y_ref:Using y_ref in the Supervised Fine-Tuning (SFT) dataset as the regularization target.Using the best ranked candidate as the regularization target.
5. Using y_ref in the Supervised Fine-Tuning (SFT) dataset as the regularization target.
6. Using the best ranked candidate as the regularization target.

Please note that the exact choices of `y_pos`, `y_neg`, and `y_ref` would depend on the specific task and data, and these are just the approaches discussed in the paper (Page 3-5).

### You
How are the feedback models trained? 

### ChatGPT
The feedback models, which include both the ranking model and the reward model, are trained on the human feedback data (DHF). Here's how it's done:

1. Ranking Model: The ranking model is trained to predict the preference between two summaries given the same document. The training data for the ranking model consists of pairs of summaries (y0, y1) for each document x, along with an indicator i which is either 0 or 1, indicating whether y0 or y1 is the preferred summary according to human feedback. The ranking model is trained to predict this preference.
2. Reward Model: The reward model is trained to assign a reward score to a summary given a document. The training data for the reward model is derived from the human feedback data by binarizing each ranked pair into a positive and a negative sequence. The reward model is trained to assign a higher score to the positive sequence than the negative sequence.

The models are trained on the training split of the human feedback data, and the best checkpoints are selected based on their performance on the validation split of the human feedback data. The ranking model and the reward model are both T5-XXL models, and they are trained with a batch size of 128 and a learning rate of 10^-3 (Page 4).

Please note that the exact training procedure might involve additional steps and considerations, such as handling of out-of-distribution decodes, and these are just the main steps as described in the paper.

### You
Which datasets does the paper use?

### ChatGPT
The paper uses the Reddit TL;DR summarization datasets from Stiennon et al. [18]. The dataset contains both fine-tune data (DSFT) and human feedback data (DHF), along with their SFT (Supervised Fine-Tuning) and RLHF (Reinforcement Learning from Human Feedback) model decodes which are used for comparison with the models proposed in the paper.

The DSFT is a filtered version of the Reddit TL;DR dataset [19]. It contains 117k/6k/6k examples in train, validation, and test splits. The DHF consists of 64k human preferences on decodes from multiple models (Page 4).

### You
If the model is GPT like, then how to back prop through this?

model(x, y_pos).log_prob(y_pos)

### ChatGPT
In a GPT-like model, the model's forward pass generates a sequence of tokens, and the `log_prob` method computes the log probabilities of these tokens. The model's parameters are used in both the forward pass and the `log_prob` computation, so gradients can be backpropagated through both of these operations to update the model's parameters.

Here's a simplified example of how this might look in PyTorch:

```python
# Forward pass
output = model(x, y_pos)

# Compute the loss
loss = -output.log_prob(y_pos).mean()

# Backward pass
loss.backward()

# Update the model's parameters
optimizer.step()

```

In this example, `model(x, y_pos)` performs the forward pass of the model, generating a sequence of tokens. The `log_prob` method then computes the log probabilities of the actual tokens `y_pos` given the input `x` and the preceding tokens. The negative mean of these log probabilities is used as the loss, which is then backpropagated through the model's parameters to compute the gradients. Finally, the optimizer updates the model's parameters based on these gradients.

Please note that this is a simplified example and might not work out-of-the-box depending on your specific model and data. For example, this code assumes that the model's `log_prob` method computes the log probabilities of a sequence given an input, and that the sequences `x` and `y_pos` are already properly encoded and aligned. You might need to adjust this code to fit your specific needs.

### You
So, we can train token by token? 

### ChatGPT
Yes, in autoregressive models like GPT, training is typically done token by token. This is because the prediction of each token in the sequence is dependent on all previous tokens.

During training, the model is given an input sequence and is trained to predict the next token in the sequence. The loss function is calculated based on the difference between the model's predictions and the actual next tokens in the sequence. This loss is then backpropagated through the model to update its parameters.

Here's a simplified example of how this might look in PyTorch:

```python
# Assume x is the input sequence and y is the target sequence
for i in range(1, len(y)):
    # The input sequence is the start of the target sequence up to the current token
    input_sequence = y[:i]
    
    # The target token is the current token in the target sequence
    target_token = y[i]
    
    # Forward pass through the model
    output = model(x, input_sequence)
    
    # Compute the loss based on the model's prediction and the target token
    loss = loss_function(output, target_token)
    
    # Backward pass
    loss.backward()
    
    # Update the model's parameters
    optimizer.step()

```

In this example, the model is trained to predict each token in the target sequence `y` based on the input sequence `x` and all previous tokens in `y`. The loss is computed for each token prediction, and the model's parameters are updated based on this loss.

Please note that this is a simplified example and might not work out-of-the-box depending on your specific model and data. You might need to adjust this code to fit your specific needs.

### You
Can you rewrite the def slic_loss(model, x, y_pos, y_neg, y_ref, delta, lambda_reg) function so that it is GPT-like model compatible? 

### ChatGPT
Sure, here's an updated version of the `slic_loss` function that should be compatible with GPT-like models. This version assumes that the model's forward method returns the logits for each token in the vocabulary, and that the sequences are encoded as integer tensors where each integer represents a token.

```python
import torch
import torch.nn.functional as F

def slic_loss(model, x, y_pos, y_neg, y_ref, delta, lambda_reg):
    """
    SLiC-HF loss function.

    Args:
        model: The model being trained.
        x: The input sequence.
        y_pos: The positive sequence.
        y_neg: The negative sequence.
        y_ref: The target sequence.
        delta: The margin for the ranking loss.
        lambda_reg: The regularization weight.

    Returns:
        The computed loss.
    """
    # Forward pass through the model for each sequence
    logits_pos = model(x, y_pos)
    logits_neg = model(x, y_neg)
    logits_ref = model(x, y_ref)

    # Calculate the log probabilities of the sequences
    log_prob_y_pos = F.log_softmax(logits_pos, dim=-1).gather(1, y_pos.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    log_prob_y_neg = F.log_softmax(logits_neg, dim=-1).gather(1, y_neg.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    log_prob_y_ref = F.log_softmax(logits_ref, dim=-1).gather(1, y_ref.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    # Calculate the calibration loss
    calibration_loss = F.relu(delta - log_prob_y_pos + log_prob_y_neg)

    # Calculate the regularization loss
    regularization_loss = lambda_reg * log_prob_y_ref

    # Combine the losses
    loss = calibration_loss - regularization_loss

    return loss.mean()

```

This version of the function uses the `gather` method to pick out the logits of the actual tokens in each sequence, and then applies the `log_softmax` function to these logits to get the log probabilities. The `unsqueeze` and `squeeze` methods are used to adjust the shapes of the tensors for these operations. The loss is then calculated as before, and the mean loss across the batch is returned.

Please note that this is a simplified example and might not work out-of-the-box depending on your specific model and data. You might need to adjust this code to fit your specific needs.

### You
Since, there is no reward function, we do not need the whole sequence before doing a backprop step.

### ChatGPT
Yes, you're correct. In the context of SLiC-HF, the loss function is based on the log probabilities of the sequences, not on a reward function. This means that you can compute the loss and perform a backpropagation step after each token is generated, rather than waiting for the entire sequence to be generated. This is often referred to as "online" or "step-by-step" training.

Here's a simplified example of how this might look in PyTorch:

```python
# Assume x is the input sequence, y_pos and y_neg are the target sequences
for i in range(1, len(y_pos)):
    # The input sequence is the start of the target sequence up to the current token
    input_sequence_pos = y_pos[:i]
    input_sequence_neg = y_neg[:i]

    # Forward pass through the model
    output_pos = model(x, input_sequence_pos)
    output_neg = model(x, input_sequence_neg)

    # Compute the loss
    loss = slic_loss(model, x, output_pos, output_neg, y_ref, delta, lambda_reg)

    # Backward pass
    loss.backward()

    # Update the model's parameters
    optimizer.step()

```

In this example, the model is trained to predict each token in the target sequences `y_pos` and `y_neg` based on the input sequence `x` and all previous tokens in `y_pos` and `y_neg`. The loss is computed for each token prediction, and the model's parameters are updated based on this loss.

Please note that this is a simplified example and might not work out-of-the-box depending on your specific model and data. You might need to adjust this code to fit your specific needs.

### You
Could you explain the section "SLiC-HF with Sample and Rank"

### ChatGPT
The "SLiC-HF with Sample and Rank" section of the paper describes one of the approaches for implementing the Sequence Likelihood Calibration from Human Feedback (SLiC-HF) method.

In this approach, candidates are sampled from the Supervised Fine-Tuning (SFT) model's training split. These candidates are then ranked using one of two models trained on the human feedback data:

1. Pointwise Reward Model: This model is trained by binarizing each ranked pair into a positive and a negative sequence. The input sequences are formatted as '[Context] ... [Summary] ...' and the target sequences are either 'Good' or 'Bad'. At inference time, the model computes the probability of the token 'Good' on the decoder side to score each of the candidates in a list, and m positive/negative pairs are sampled from them.
2. Pairwise Ranking Model: This model is trained by formulating the human feedback into a pairwise ranking problem. The input sequences are formatted as '[Context] ... [Summary A] ... [Summary B]' and the target sequences are either 'A' or 'B'. At inference time, a tournament-style procedure is used to rank the candidates in a list. For example, given a list of 4 candidates c1, c2, c3, c4, the model first ranks c1, c2 and c3, c4 and then ranks winner(c1, c2), winner(c3, c4). Given m candidates, the ranking model is called m - 1 times and m - 1 positive/negative pairs are yielded.

The paper also discusses the choice of the target sequence `y_ref` for cross-entropy regularization. Two options are considered: using `y_ref` in the Supervised Fine-Tuning (SFT) dataset as the regularization target, or using the best-ranked candidate as the regularization target.

The advantage of the "SLiC-HF with Sample and Rank" approach is that it leverages the ranking or reward model to determine the positive and negative sequences, which aligns well with the task of learning from human feedback. However, it also involves additional complexity and computational cost due to the need to decode from the SFT model and train a separate model to label the decodes.

### You
In SLIC ablation section, what does "we also observe an increase in average length of models" mean?

### ChatGPT
The "increase in average length of models" in the SLiC ablation section refers to the observation that the summaries generated by the models trained with SLiC-HF (Sequence Likelihood Calibration from Human Feedback) tend to be longer on average compared to those generated by the baseline models.

This observation is made in the context of comparing different methods for leveraging human feedback data. The authors note that while the ROUGE scores (a common metric for evaluating the quality of text summaries) tend to drop when learning from human feedback (since there's less incentive for the generated summaries to be similar to the reference texts), the average length of the generated summaries tends to increase.

The authors also conduct a length-controlled study to account for this increase in length. In this study, they calculate the average quality of the summaries generated by the models for different lengths, and compare these quality scores to those of the baseline models. This allows them to evaluate the performance of the models in a way that's not biased by the length of the generated summaries.

The increase in average length of the summaries is likely due to the fact that the models are being trained to maximize the likelihood of the human feedback, which might favor longer summaries. However, the exact reasons for this increase in length would depend on the specifics of the human feedback data and the training process.

### You
What ablations are done in the "continue to finetune" section? 

### ChatGPT
In the "Continue to Fine-tune" section, the authors conduct several ablation experiments to compare different methods of leveraging human feedback data. Here are the ablations done:

1. Continue Fine-tuning on Filtered Data: The authors consider three approaches to filter data for continued fine-tuning:Keep only positive human feedback sequences and discard negative ones.Decode 8 summaries from the Supervised Fine-Tuning (SFT) model, use the ranking model to select the best 1 out of 8 summaries by a tournament-style ranking approach.Decode 8 summaries from the SFT model, use the reward model to select the best 1 out of 8 summaries by scoring each and taking the one with the max score.
2. Keep only positive human feedback sequences and discard negative ones.
3. Decode 8 summaries from the Supervised Fine-Tuning (SFT) model, use the ranking model to select the best 1 out of 8 summaries by a tournament-style ranking approach.
4. Decode 8 summaries from the SFT model, use the reward model to select the best 1 out of 8 summaries by scoring each and taking the one with the max score.
5. Apply SLiC-HF Directly On Human Feedback Data (SLiC-HF-direct): The authors observe that while the calibration loss decreases as expected, the sequence length keeps increasing and does not converge to a stable value. They hypothesize that SLiC-HF-direct is prone to out-of-distribution decodes generated by other models in the human feedback data. However, when using the ranking model to select for the best checkpoint for SLiC-HF-direct, it has moderate length increment and has a win rate against reference close to SLiC-HF-sample-rank.
6. Apply SLiC-HF on Ranked Model Decodes (SLiC-HF-sample-rank): The authors find that SLiC-HF-sample-rank using the ranking model has about a 3% gain in win rate against reference compared to SLiC-HF-sample-rank using the reward model. They also find that for SLiC-HF-sample-rank using the ranking or the reward model, using SFT targets or best ranked decodes as regularization doesn’t show much

### You
How many GPU hours were used?

### ChatGPT
The document does not explicitly mention the number of GPU hours used for the experiments. The paper discusses the compute and memory efficiency differences between SLiC-HF and RLHF-PPO, but it does not provide specific numbers in terms of GPU hours.

### You
Does the ranking model run alongside the main model or does it run offline?

### ChatGPT
The ranking model in the SLiC-HF (Sequence Likelihood Calibration from Human Feedback) approach is trained separately and runs offline. It is not trained alongside the main model.

In the SLiC-HF approach, the ranking model is used to rank the candidate summaries that are generated by the main model. These rankings are then used to determine the positive and negative sequences for the SLiC-HF loss function.

Once the ranking model has been trained, it can be used to rank new candidate summaries without needing to be retrained or updated during the training of the main model. This allows the ranking model to be run offline, which can be more computationally efficient and can also simplify the training process.

### You
So, the steps are these?

1. Summary model is trained on data
2. Summary model generates more data
3. Ranking model ranks them and adds back to the training corpus
4. Go to 1



### ChatGPT
Yes, you've got the general idea, but let's clarify a few points to ensure we're on the same page. Here's a more detailed breakdown:

1. Train the Supervised Fine-Tuning (SFT) model: This is the initial summary model, and it's trained on the available dataset.
2. Generate candidate summaries: Use the trained SFT model to generate candidate summaries for the documents in the dataset.
3. Collect human feedback: Human evaluators rank the candidate summaries based on their quality. This forms the human feedback data (DHF).
4. Train the ranking model: The ranking model is trained on the human feedback data to predict the rankings of the candidate summaries.
5. Rank the candidate summaries: Use the trained ranking model to rank new candidate summaries generated by the SFT model. The ranking model assigns a score to each candidate summary, and these scores are used to determine the positive and negative sequences for the SLiC-HF loss function.
6. Train the main model with SLiC-HF: The main model (which could be the same as or different from the SFT model) is then trained using the SLiC-HF loss function, which uses the rankings from the ranking model.
7. Repeat steps 2-6: This process can be repeated to continually improve the main model. In each iteration, the main model should generate better candidate summaries, leading to better human feedback data and better rankings from the ranking model.

Please note that this is a simplified overview of the process described in the paper, and the actual process might involve additional steps and considerations. For example, the paper also discusses different methods for selecting the positive and negative sequences and the regularization target, as well as different methods for leveraging the human feedback data.

### You
Does the paper mention how many times was it repeated? 

### ChatGPT
The paper does not explicitly mention how many times the process of generating summaries, collecting human feedback, training the ranking model, and retraining the main model was repeated. The authors describe the process and the results of their experiments, but they do not provide specific numbers for the number of iterations.

However, they do mention that they conducted several ablation studies and comparisons to evaluate different methods of leveraging the human feedback data and different configurations of the SLiC-HF approach. These experiments likely involved multiple iterations of the process, but the exact number is not specified in the paper.

### You
Can you think of any more ablations the authors should have performed?

### ChatGPT
The authors of the paper have performed a comprehensive set of ablations and experiments to evaluate the performance of the SLiC-HF approach. However, there are always additional experiments that could potentially provide further insights. Here are a few possibilities:

1. Different Ranking Models: The authors used a pointwise reward model and a pairwise ranking model to rank the candidate summaries. It could be interesting to evaluate the performance of the SLiC-HF approach with different types of ranking models, such as listwise ranking models or models based on different ranking algorithms.
2. Different Regularization Targets: The authors considered using the SFT targets and the best-ranked decodes as the regularization target. Additional experiments could be conducted to evaluate the performance of the SLiC-HF approach with different choices of regularization targets.
3. Different Model Architectures: The authors used a T5 model for their experiments. It could be interesting to evaluate the performance of the SLiC-HF approach with different model architectures, such as Transformer-XL, BERT, or GPT-3.
4. Different Datasets: The authors used the Reddit TL;DR dataset for their experiments. Additional experiments could be conducted to evaluate the performance of the SLiC-HF approach on different datasets, such as news articles, scientific papers, or books.
5. Hyperparameter Tuning: The authors used a fixed set of hyperparameters for their experiments. Additional experiments could be conducted to evaluate the performance of the SLiC-HF approach with different hyperparameters, such as different learning rates, batch sizes, or regularization weights.
6. Long-term Stability: The authors observed that the sequence length kept increasing and did not converge to a stable value when applying SLiC-HF directly on human feedback data. It could be interesting to conduct additional experiments to investigate the long-term stability of the SLiC-HF approach, such as by running the training process for a larger number of iterations or by using different methods to control the sequence length.

Please note that these are just suggestions, and the feasibility and usefulness of these additional experiments would depend on the specific goals of the research and the available resources.

### You
minuts between 11:08 AM and 8:17 AM

### ChatGPT
To calculate the minutes between 11:08 AM and 8:17 AM, you need to consider that 8:17 AM is on the next day because it's earlier than 11:08 AM.

First, calculate the minutes from 11:08 AM to 12:00 PM (noon):

```scss
60 minutes (in an hour) - 8 minutes = 52 minutes

```

Then, calculate the minutes from 12:00 AM (midnight) to 8:17 AM:

```scss
8 hours * 60 minutes (in an hour) + 17 minutes = 497 minutes

```

Finally, add these two results together:

```
52 minutes + 497 minutes = 549 minutes

```

So, there are 549 minutes between 11:08 AM and 8:17 AM the next day.

### You
### What problem does it solve?

### How does it solve it?

#### Dataset

#### Model

### How is this paper novel?

### Key takeaways

### What I still do not understand?

### Ideas to pursue

### ChatGPT
What problem does it solve?

The paper addresses the problem of improving the quality of text summaries generated by machine learning models. The authors focus on the challenge of effectively leveraging human feedback data to train these models, with the goal of producing summaries that are more aligned with human preferences.

How does it solve it?

The authors propose a new method called Sequence Likelihood Calibration from Human Feedback (SLiC-HF). This method involves training a model to generate candidate summaries, collecting human feedback data by having human evaluators rank these summaries, and then using this feedback data to train a ranking model. The ranking model is used to determine the positive and negative sequences for the SLiC-HF loss function, which is used to train the main model.

Dataset

The authors use the Reddit TL;DR dataset for their experiments. This dataset consists of Reddit posts and their corresponding summaries, which are provided by the authors of the posts.

Model

The authors use a T5 model for their experiments. They also train a separate ranking model to rank the candidate summaries based on the human feedback data.

How is this paper novel?

The novelty of this paper lies in the proposed SLiC-HF method, which introduces a new way of leveraging human feedback data to improve the quality of text summaries generated by machine learning models. The authors also conduct a comprehensive set of experiments and ablation studies to evaluate the performance of the SLiC-HF method and compare it with other methods.

Key takeaways

The key takeaway from this paper is that the SLiC-HF method can effectively leverage human feedback data to improve the quality of text summaries generated by machine learning models. The authors find that the SLiC-HF method outperforms the baseline models in their experiments, and they also observe that the summaries generated by the models trained with SLiC-HF tend to be longer on average.

What I still do not understand?

The paper is quite comprehensive, but there might be some aspects that require further clarification. For example, the exact process of training the ranking model and determining the positive and negative sequences for the SLiC-HF loss function might not be entirely clear. Additionally, the reasons for the observed increase in the average length of the summaries might warrant further investigation.

Ideas to pursue

One potential idea to pursue is to apply the SLiC-HF method to other types of text generation tasks, such as translation or dialogue generation. It could also be interesting to experiment with different types of ranking models or different methods for collecting human feedback data. Additionally, further research could be conducted to investigate the reasons for the observed increase in the average length of the summaries and to develop methods for controlling this length.
