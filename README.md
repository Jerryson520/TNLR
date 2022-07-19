# Microsoft Turing Academic Program

As part of Microsoft’s [AI at Scale](https://innovation.microsoft.com/en-us/ai-at-scale) initiative, we developed a family of large AI models called the Microsoft Turing models. The Turing models are trained with billions of pages of publicly available text, absorbing the nuances of language, grammar, knowledge, concepts and context. The models excel at multiple language tasks, such as completing predictions, reading comprehension, common sense reasoning, paraphrasing a lengthy speech, finding relevant passages across thousands of documents, or word-sense disambiguation.  

We created the Microsoft Turing Academic Program (MS-TAP) as part of our commitment to democratizing AI, and the responsible development of Microsoft’s AI models. The program provides access to Turing models in order to support high-impact scholarly research, including efforts aimed at advancing principles of learning and reasoning, exploring novel applications, and pursuing better understanding of challenges and opportunities with regard to the ethical and responsible use of large-scale neural models.  

These models have already been used to improve different language understanding tasks across Bing, Office, Dynamics, Edge and other productivity products. In Bing they are being used for caption generation, and [question answering](https://blogs.bing.com/search/2020_05/AI-at-Scale-in-Bing) and summarization across 100+ languages.  Across Office, Turing models are powering features that include [Text Prediction](https://insider.office.com/en-us/blog/text-predictions-in-word-outlook) in Word, Outlook and Teams, to help users type faster and with fewer mistakes; [Suggested replies](https://www.microsoft.com/en-us/research/group/msai/articles/assistive-ai-makes-replying-easier-2/) in Outlook to automatically recommends a response to an email and in Word; and [Smart Find](https://insider.office.com/en-us/blog/microsoft-search-search-your-document-like-you-search-the-web) to enable a much broader set of search queries beyond “exact match.” In Dynamics, Turing has been adapted with business domain data using Azure technologies, which are fully compliant with user privacy and enterprise contractual obligations. Applied within Dynamics 365 Sales Insights the next best action with a customer is suggested based on previous interactions. 

## Blog Posts
* [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, the World’s Largest and Most Powerful Generative Language Model](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)
* [Microsoft Turing Universal Language Representation model, T-ULRv5, tops XTREME leaderboard and trains 100x faster](https://www.microsoft.com/en-us/research/blog/microsoft-turing-universal-language-representation-model-t-ulrv5-tops-xtreme-leaderboard-and-trains-100x-faster/)  
* [Turing Bletchley: A Universal Image Language Representation model by Microsoft - Microsoft Research](https://www.microsoft.com/en-us/research/blog/turing-bletchley-a-universal-image-language-representation-model-by-microsoft/) 
* [Microsoft Makes It Easier To Build Popular Language Representation Model At Large Scale](https://azure.microsoft.com/en-us/blog/microsoft-makes-it-easier-to-build-popular-language-representation-model-bert-at-large-scale/) 
* [How Azure Machine Learning Powers SuggestedReplies In Outlook](https://azure.microsoft.com/en-us/blog/how-azure-machine-learning-service-powers-suggested-replies-in-outlook/) 
* [Better Document Previews using the Microsoft Turing Model for Natural Language Representations - Microsoft Research](https://www.microsoft.com/en-us/research/group/msai/articles/better-document-previews-using-the-microsoft-turing-model-for-natural-language-representations/) 
* [How Azure Machine Learning Enables Powerpoint Designer](https://azure.microsoft.com/en-us/blog/how-azure-machine-learning-enables-powerpoint-designer/) 
* [Accelerate Your NLP Pipelines Using Huggingrface Transformers & ONNX Runtime](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333) 
* [OpenAI - Azure Supercomputer](https://blogs.microsoft.com/ai/openai-azure-supercomputer/) 
* [Microsoft details how it improved Bing's autosuggest recommendations with AI](https://venturebeat.com/2020/09/23/microsoft-details-how-it-improved-bings-autosuggest-recommendations-with-ai/) 

## Turing Natural Language Representation Model (T-NLRv5) Overview
We are excited to release a private preview of the Turing Natural Language Representation v5 (TNLRv5) model to our MS-TAP partners as part of our commitment to responsible AI development. MS-TAP partners will have access to the base (12-layer, 768 hidden, 12 attention heads, 184M parameters) and large (24-layer, 1024 hidden, 16 attention heads, 434M parameters) T-NLRv5 model.   

T-NLRv5 integrates some of the best modeling techniques developed by Microsoft Research, Azure AI, and Microsoft Turing. The models are pretrained at large scale using an efficient training framework based on FastPT and DeepSpeed. T-NLRv5 is the state of the art at the top of SuperGLUE and GLUE leaderboards, further surpassing human performance and other models. Notably, T-NLRv5 first achieved human parity on MNLI and RTE on the GLUE benchmark, the last two GLUE tasks which human parity had not yet met. In addition, T-NLRv5 is more efficient than recent pretraining models, achieving comparable effectiveness with 50% fewer parameters and pretraining computing costs. 

T-NLRv5 is largely based on our recent work, [COCO-LM](https://arxiv.org/abs/2102.08473), a natural evolution of pretraining paradigm converging the benefits of ELECTRA-style models and corrective language model pretraining. Read more about TNLRv5 in our [blog post](https://www.microsoft.com/en-us/research/blog/efficiently-and-effectively-scaling-up-language-model-pretraining-for-best-language-representation-model-on-glue-and-superglue/).  

## Training Data and Model Configuration
This model employs an auxiliary transformer language model to corrupt an input text sequence, and the main transformer model is pretrained using the corrective language model task, which is to detect and correct tokens replaced by the auxiliary model. This augments the ELECTRA model family with language modeling capacity, bringing together the benefits from pretraining with adversarial signals generated from the auxiliary model and the language modeling capacity, which is handy for prompt-based learning. 

## Guidelines
Like other publicly available language models, the Microsoft Turing models are trained with billions of pages of publicly available text, and hence may have picked up biases around gender, race and more from these public documents. Mitigating negative effects from these biases is a hard, industry-wide issue and Microsoft is committed to the advancement and use of AI grounded in principles that put people first and benefit society. We are putting these [Microsoft AI principles](https://www.microsoft.com/en-us/ai/responsible-ai?activetab=pivot1%3aprimaryr6) into practice throughout the company and have taken an extensive number of precautionary measures to prevent these implicit biases getting exhibited when using the models in our products. We strongly encourage developers to do the same by putting appropriate guardrails and mitigations in place before taking these models to production. [Learn more](https://github.com/microsoft/Turing-Academic/blob/main/Transparency_Note_Microsoft_Turing_NLR.pdf) about the Microsoft Turing language models limitations and risks.

## Structure of repository
**1. configs**
   * tnlrv5-base-cased.json: configuration for base model.
   * tnlrv5-large-cased.json: configuration for large model.
  
**2. models**
   * tnlrv5_base.pt: checkpoint for base model.
   * tnlrv5_large.pt: checkpoint for large model.
  
**3. src**
* tnlrv5: containing utility functions for tnlrv5 model
* GLUE
  * download_glue_data.py: download GLUE data.
  * run_classifier.py: main body of GLUE finetuning task.
  * utils_for_glue.py: utility functions of glue datasets.
* SQuAD
  * run_squad.py: main body of SQuAD finetuning task.
  * utils_for_squad.py: utility functions of squad dataset.
  * utils_squad_evaluate.py: utility functions of evaluating squad dataset.

**4. vocab**
* dict.txt: dictionary of vocabulary
* sp.model


## Model Setup (On VMware)
1. Install *git lfs*:
   ```bash
   apt-get update
   apt-get install git-lfs
   ```

2. *Clone the repository* using personal token:  
   * Personal token:
      Create a personal token following [this](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) link.  

   * Clone the repository
      ```bash
      git lfs clone https://username@github.com/username/mstap-TNLR-harvard-cai-lu
      ```
      [Password: Personal token]
   
3. *Pytorch* >= 1.6, CUDA version
   * Installing PyTorch following [this](https://pytorch.org/get-started/previous-versions/) link

4. *Apex* (same version as CUDA PyTorch)
   * Install apex following instruction [here](https://github.com/NVIDIA/apex).
   * Apex will be installed successfully if:
      ```bash
      runtime api version: nvcc -V
      ```
      ```python
      import torch
      print(torch.__version__)
      ```
      are the same.


5. *Transformers*
   Install transformers using:
   ```bash
   pip install transformers == 2.10.0 
   ```    

## GLUE Finetuning 
### Downloading the GLUE dataset
The [GLUE dataset](https://gluebenchmark.com/tasks) can be downloaded by running the following script

```shell
# Set path for this repository
export HOME_DIR=~/mstap-TNLR-harvard-cai-lu
cd ${HOME_DIR}
python src/download_glue_data.py
```

### MNLI (base)
```shell
 # Set path to read training/dev dataset that was downloaded in the previous step
export DATASET_PATH=${HOME_DIR}/glue_data/MNLI

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/mnli_base_ft/

export TASK_NAME=mnli

# Set model name (or checkpoint path) for finetuning 
export CKPT_PATH=tnlrv5-base-cased

# Set max sequence length
export MAX_LEN=512

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-base-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_base_cased.$MAX_LEN.cache
export DEV_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_base_cased.$MAX_LEN.cache

# Setting the hyperparameters for the run
export BSZ=32
export LR=1e-5
export EPOCH=5
export WD=0.1
export WM=0.0625

CUDA_VISIBLE_DEVICES=0 python src/run_classifier.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH --task_name $TASK_NAME \
    --data_dir $DATASET_PATH --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --do_train --evaluate_during_training --logging_steps 1000 --output_dir $OUTPUT_PATH --max_grad_norm 0 \
    --max_seq_length $MAX_LEN --per_gpu_train_batch_size $BSZ --learning_rate $LR \
    --num_train_epochs $EPOCH --weight_decay $WD --warmup_ratio $WM \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --dropout_prob 0.1 --cls_dropout_prob 0.1 \
    --fp16 --fp16_opt_level O2 --seed 1
 ```

#### Sample results (Acc):

`--seed`: 1

 ```
MNLI-m: 90.219
MNLI-mm: 90.155
```

### MNLI (large)
#### 1. Single GPU
```shell
 # Set path to read training/dev dataset that was downloaded in the previous step
export DATASET_PATH=${HOME_DIR}/glue_data/MNLI

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/mnli_large_ft/

export TASK_NAME=mnli

# Set model name (or checkpoint path) for finetuning 
export CKPT_PATH=tnlrv5-large-cased

# Set max sequence length
export MAX_LEN=512

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-large-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_large_cased.$MAX_LEN.cache
export DEV_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_large_cased.$MAX_LEN.cache

# Setting the hyperparameters for the run.
export BSZ=32
export LR=3e-6
export EPOCH=2
export WD=0.1
export WM=0.0625
CUDA_VISIBLE_DEVICES=0 python src/run_classifier.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH --task_name $TASK_NAME \
    --data_dir $DATASET_PATH --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --do_train --evaluate_during_training --logging_steps 1000 --output_dir $OUTPUT_PATH --max_grad_norm 0 \
    --max_seq_length $MAX_LEN --per_gpu_train_batch_size $BSZ --learning_rate $LR \
    --num_train_epochs $EPOCH --weight_decay $WD --warmup_ratio $WM \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --dropout_prob 0.1 --cls_dropout_prob 0.1 \
    --fp16 --fp16_opt_level O2 --seed 1
 ```
 
 #### Sample results (Acc):

`--seed`: 1

 ```
MNLI-m: 不太清楚需不需要这个model需不需要分成single GPUs 和 multiple GPUs
MNLI-mm: 
```


#### 2. Multiple GPUs
```shell
# Set path to read training/dev dataset that was downloaded in the previous step
export DATASET_PATH=${HOME_DIR}/glue_data/MNLI

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/mnli_large_ft/

export TASK_NAME=mnli

# Set model name (or checkpoint path) for finetuning 
export CKPT_PATH=tnlrv5-large-cased

# Set max sequence length
export MAX_LEN=512

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-large-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_large_cased.$MAX_LEN.cache
export DEV_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_large_cased.$MAX_LEN.cache

# Setting the hyperparameters for the run.
# per_gpu_train_batch_size = train_batch_size / num_gpus = 32 / 8 = 4
export BSZ=4
export LR=3e-6
export EPOCH=2
export WD=0.1
export WM=0.0625
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 src/run_classifier.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH --task_name $TASK_NAME \
    --data_dir $DATASET_PATH --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --do_train --evaluate_during_training --logging_steps 1000 --output_dir $OUTPUT_PATH --max_grad_norm 0 \
    --max_seq_length $MAX_LEN --per_gpu_train_batch_size $BSZ --learning_rate $LR \
    --num_train_epochs $EPOCH --weight_decay $WD --warmup_ratio $WM \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --dropout_prob 0.1 --cls_dropout_prob 0.1 \
    --fp16 --fp16_opt_level O2 --seed 1
 ```

 #### Sample results (Acc):


## **SQuAD 2.0 Fine-tuning (Base & Large)**
[Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. 


### 1. Single GPU
 ```shell
# Set path for this repository
export HOME_DIR=~/mstap-TNLR-harvard-cai-lu
cd ${HOME_DIR}
# Set path to the location where the data will be downloaded
export DATASET_PATH=${HOME_DIR}/squad_data/

# Download the train & dev datset
mkdir -p ${DATASET_PATH}
# Train datset
export TRAIN_FILE=${DATASET_PATH}/train-v2.0.json
wget -O $TRAIN_FILE https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
# Dev datset
export DEV_FILE=${DATASET_PATH}/dev-v2.0.json
wget -O $DEV_FILE https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/squad_ft/

# Set path to the model checkpoint you need to test 
export CKPT_PATH=tnlrv5-base-cased

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-base-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${TRAIN_FILE}_tnlrv5_base_cased.384doc.cache
export DEV_CACHE=${DEV_FILE}_tnlrv5_base_cased.384doc.cache

# Setting the hyperparameters for the run.
export BSZ=32
export LR=3e-5
export EPOCH=3
CUDA_VISIBLE_DEVICES=0 python src/run_squad.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --train_file $TRAIN_FILE --predict_file $DEV_FILE \
    --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --do_train --do_eval \
    --per_gpu_train_batch_size $BSZ --learning_rate $LR --num_train_epochs $EPOCH --gradient_accumulation_steps 1 \
    --max_seq_length 384 --doc_stride 128 --output_dir $OUTPUT_PATH \
    --version_2_with_negative --seed 1 --max_grad_norm 0 \
    --weight_decay 0.1 --warmup_ratio 0.0625 \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --fp16_opt_level O2 --fp16
```
`--seed`: 1

```
F1_score: 88.207
```


### 2. Multiple GPUs
```shell
# Set path for this repository
export HOME_DIR=~/mstap-TNLR-harvard-cai-lu
cd ${HOME_DIR}
# Set path to the location where the data will be downloaded
export DATASET_PATH=${HOME_DIR}/squad_data/

# Download the train & dev datset
mkdir -p ${DATASET_PATH}
# Train datset
export TRAIN_FILE=${DATASET_PATH}/train-v2.0.json
wget -O $TRAIN_FILE https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
# Dev datset
export DEV_FILE=${DATASET_PATH}/dev-v2.0.json
wget -O $DEV_FILE https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/squad_ft/

# Set path to the model checkpoint you need to test 
export CKPT_PATH=tnlrv5-base-cased

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-base-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${TRAIN_FILE}_tnlrv5_base_cased.384doc.cache
export DEV_CACHE=${DEV_FILE}_tnlrv5_base_cased.384doc.cache

# Setting the hyperparameters for the run.
# per_gpu_train_batch_size = train_batch_size / num_gpus = 32 / 8 = 4
export BSZ=4
export LR=3e-5
export EPOCH=3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 src/run_squad.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --train_file $TRAIN_FILE --predict_file $DEV_FILE \
    --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --do_train --do_eval \
    --per_gpu_train_batch_size $BSZ --learning_rate $LR --num_train_epochs $EPOCH --gradient_accumulation_steps 1 \
    --max_seq_length 384 --doc_stride 128 --output_dir $OUTPUT_PATH \
    --version_2_with_negative --seed 1 --max_grad_norm 0 \
    --weight_decay 0.1 --warmup_ratio 0.0625 \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --fp16_opt_level O2 --fp16
 ```


## Papers
* [COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining](https://arxiv.org/abs/2102.08473)
* [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)
* [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
## Contributing 
See [CONTRIBUTING.md](./CONTRIBUTING.md). 

## License 
See [LICENSE.txt](./LICENSE.txt). 
## Security 
See [SECURITY.md](./SECURITY.md). 

## Support 
Please email us at turing-academic@microsoft.com for troubleshooting, or file an issue through the repo