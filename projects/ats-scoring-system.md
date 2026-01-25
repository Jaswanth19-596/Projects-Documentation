# ATS Scoring System

**Status:** active
**Started:** Jan 20, 2026, 03:43 PM
**Total Time:** 0m

## Overview
An AI model that takes in the resume and job description as input and outputs an ATS score - a number between 0 - 100 that indicates how well the resume matches the job description.

The model not only does the simple keyword matching, it will also take into consideration:
1. Semantic similarity (different words, same meaning).
2. Context and relevance.
3. Skills alignment.
4. Experience matching.
5. Overall fit.

## Initial Approach
Instead of using a LLM, I'm planning to use a smaller transformer model and finetune it against a relevant dataset to predict the ATS score.

**Technologies:** Transformers, Fine Tuning.

## Project Timeline

### 1. Finding a Dataset
*Jan 20, 2026, 03:45 PM* | Category: research

Found a dataset on Hugging face - 0xnbk/resume-ats-score-v1-en which has the following features:
- It has train, test sets.
- It has the resume, job description and ats score features.

### 2. Splitting the training set into train and validation split.
*Jan 20, 2026, 03:46 PM* | Category: coding

**Resources used:**
- [ATS_score_generator.ipynb - Colab](https://colab.research.google.com/drive/1sYhOGmLPHr1uigRWtSicu4TGtpwD0UbO#scrollTo=e_Ne7WR-Few2)

### 3. Preprocessing the dataset
*Jan 20, 2026, 03:47 PM* | Category: coding

(Renaming columns, removing null and duplicate values)

**Resources used:**
- [ATS_score_generator.ipynb - Colab](https://colab.research.google.com/drive/1sYhOGmLPHr1uigRWtSicu4TGtpwD0UbO#scrollTo=e_Ne7WR-Few2)
- [ATS_score_generator.ipynb - Colab](https://colab.research.google.com/drive/1sYhOGmLPHr1uigRWtSicu4TGtpwD0UbO#scrollTo=ZmPuZn2-qDqk)

### 4. Loading the tokenizer and model
*Jan 20, 2026, 03:50 PM* | Category: coding

I'm planning to use the following model: 

- google-bert/bert-base-uncased

**Resources used:**
- [Claude](https://claude.ai/chat/0cf57086-2760-4e44-a47f-5460f42859fa)
- [Tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer)

### 5. Using the LongFormer Architecture
*Jan 20, 2026, 04:14 PM* | Category: coding

[Input Text] -> [Longformer Encoder] -> [Regression Head] -> [Score]

**Resources used:**
- [ATS_score_generator.ipynb - Colab](https://colab.research.google.com/drive/1sYhOGmLPHr1uigRWtSicu4TGtpwD0UbO#scrollTo=ZmPuZn2-qDqk)

### 6. Creating a custom class for Head of Longformer
*Jan 20, 2026, 06:46 PM* | Category: research

Why are we creating a custom head for regression:

- There is No AutoModelForSequenceRegression in HuggingFace.

### 7. Creating the Tokenize Function
*Jan 20, 2026, 08:47 PM* | Category: coding

The tokenize function should be having the tokenizer, global attention mask return the dataset after tokenization.

### 8. Creating the compute metrics function
*Jan 20, 2026, 09:10 PM* | Category: coding

It's just a simple function, which gets predictions and labels. All we gotta do is calculate all the metrics and return the dictionary of them.

### 9. Training the model
*Jan 20, 2026, 09:29 PM* | Category: coding

Ran into Cuda out of memory error.

### 10. Trained for 3 hours - No Progress
*Jan 22, 2026, 04:42 AM* | Category: coding

- R2 score is still negative.
- Not a lot of change in validation loss.

### 11. Used mean pooling
*Jan 24, 2026, 02:48 AM* | Category: coding

Used mean pooling but still the model is performing poorly.

## Approach Changes (Pivots)

### Pivot 1: Jan 20, 2026, 03:56 PM

**From:** Issue with BERTs tokenizer. The Tokenizer of BERT is only having a context window of size 512 tokens. But the input is having a median of 1600 tokens(resume + Job description) and a maximum of 4200 tokens. 

I also finetuned the model on BERT, but the following are the results:
- MSE : 1332.5.
- MAE : 26.34
- RMSE: 36.5
- R2 : -1.08

The model is performing worse than a mean model.

**To:** I'm planning to use Longformer as it has more context length (4026) tokens. 

I believe the information loss is the main reason that BERT is not able to perform well.

**Why:** Performance of BERT:
Test Set Evaluation Metrics
==================================================
MSE:  1332.5591
MAE:  26.3473
RMSE: 36.5042
R²:   -1.0854

**Trigger:** Worse performance of the BERT model.

*Tags: Performance issue, Learning moment, Technical limitation*

### Pivot 2: Jan 21, 2026, 07:31 PM

**From:** Was using 32 batch size, but running out of GPU memory

**To:** Had to use 1 batch size to fit everything in GPU.

**Why:** GPU running out of memory.

*Tags: Performance issue*

### Pivot 3: Jan 21, 2026, 07:49 PM

**From:** - Mistakes I'm doing:
    - Using a really complex regression head with limited data(3 layers).
    -  Insufficient Training:  Number of epochs = 1 => Transformer models need a lot of epochs to train.

**To:** 1. Epochs = 5
2. Increased Batch size = 2.
3. Lower learning rate = 1e-5.
4. Simple regression head.

**Why:** Poor results: R2 negative, No decrease in Validation or Training Loss.

*Tags: Performance issue, Learning moment*

### Pivot 4: Jan 21, 2026, 08:27 PM

**From:** Was using context length of size 4096 (the maximum amount of Longformer). 

Issues:
- Training is slow.
- Cuda running out of memory. 
- Only able to use batch size = 1

**To:** 90% of my data has a average of 2000 tokens per example.  So, reduced the context size to 2048 tokens. 

Improvements: 
- Training became slightly fast.
- Am able to use batch_size = 2

**Why:** Wasting a lot of memory for Padding.

## Learnings

- **skill:** BERT has a context limit of 512 tokens.

  BERT has a context limit of 512 tokens.

- **skill:** How Longformer Architecture takes the input

  :Input: "Resume: I have Python skills..." + "JD: Need Python developer..."       
↓
Tokens: [CLS] Resume: I have Python... [SEP] JD: Need Python... [SEP]
↓
IDs: [101, 2739, 1024, 1045, 2031, ...]

- **skill:** What is LongFormer

  Longformer is a cousin of BERT that can process longer documents efficiently.

- **skill:** Purpose of unpacking tokens

  Instead of writing : 

model(input_ids = tokens['input_ids'], attention_mask = tokens['attention_mask'], token_type_ids = tokens['token_type_ids'])

We write:

model(**tokens)

- **skill:** Base Model Properties

  1. outputs.last_hidden_state
2. output.pooler_output.
3. output.hidden_states.
4. output.attentions.

- **skill:** Transformers for Regression

  We can make Transformer models perform regression in two ways:

1. Creating a custom class that outputs a regression output.
2. While Loading the transformer, we can use num_labels = 1, problem_type = regression.

- **skill:** Important confusing point

  When you call the model, the model only processes one batch at a time. 

The forward method only gets one batch as input at at time.

- **skill:** Squeeze and Unsqueeze

  Squeeze : Removes unnecessary dimensions that have size 1.

Unsqueeze : Adds extra dimension of size 1 at specified position.

- **skill:** Hierarchy of the project

  1. Trainer object:
    - Needs Model.
    - Needs Dataset.
    - Needs Training args.
    - Needs Compute metrics : For measuring metrics.

2. Model:
    - Base Encoder : Longformer.
    - Regression Head: A nn.sequential.
        - Linear Layers.
        - ReLU.
        - Dropout.
        - Sigmoid.
    - Loss function (MSELoss).

3. Dataset (Should be tokenized): 
    - Train set.
    - Test set.
    - Val set.

4. Training Args:
    - Hardware settings.
    - Optimization : Learning rate, weight decay, warmup steps.
    - Strategy: eval_strategy, save_steps, logging steps.


5. Metrics: Compute Metrics:
 - Error Metrics : MSE, MAE
 - Relationship metrics : Pearson coefficient, R2 score.

- **skill:** In simple terms

  1. Take text (Resume, Job Description).
2. Convert them into tokens.
3. Make them of same length (Padding) so that you can batch them.
4. Feed to model.
5. Get a score.

- **approach:** Range of Learning Rate

  As the model size grows, learning rate keeps decreasing. 

1. Training Traditional Models:
    - Weights are randomly initialized.
    - Need bigger learning rate(0.1, 0.01) to move away from noise.
2. Fine Tuning Pretrained Transformers:
    - Weights are already trained.
    - Need small tweaks(1e-5, 1e-6) to fine tune.


---
*Generated by Project Process Documenter on Jan 25, 2026, 01:03 AM*