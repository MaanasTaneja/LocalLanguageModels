
# Local Language Modeling Library

This repository provides a **local language modeling library** built using the `transformers` library and PyTorch. It allows you to run and fine-tune language models locally on your own datasets.  Allows you to experiment with different forms of decoding, and sampling from language models, and log primitive conversation chains (in the vein of the OpenAI API and others), will in the near future also be able to fine tune 
language models using the library (FineTune.py), and will be seamless, to allow for finetuning with simple csv files with input, output pairs for instruction fine tuning, or large corpus of text for pre-training style transfer fine tuning. All complexities in converting and handling data will be solved and hidden. 

Currently, `fine_tuning.ipynb` notebook serves as a test script for fine-tuning models, has a demo of all these features (currently trains Bert to classify object and trains GP2 to output Shakespere Text). In the future, a `fine_tune.py` module will be integrated into the library to streamline the fine-tuning process.

## Features

- **Language Model Integration**: Utilize pre-trained models from the `transformers` library and fine-tune them with your own data.
- **Flexible Fine-Tuning**: Customize training parameters and datasets for fine-tuning on your local machine.
- **PyTorch Backend**: Built with PyTorch for deep learning tasks.
- **Transformers Library**: Leverage the rich functionality of the `transformers` library for model management.

## Requirements

- Python 3.7+
- `transformers` library
- `torch` (PyTorch)

You can install the necessary dependencies with:

```bash
pip install transformers torch
```

## Usage

The core file that provides functions to load, train, and use language models.

#### Example Usage

```python
model = "qwen 0.5b Instruct"
gen_model = "gpt2"
gen_init_prompt = "once upon a time there was a red fox"
params_model = {
    "device_map" : "cpu",
}
params_tokenizer = {}
system_prompt = "You are a helpful assistant."

lm_non_instruct = NonInstructionModel(gen_model, params_model, params_tokenizer, gen_init_prompt)
lm_non_instruct.prompt("once upon a time there was a red fox and he")
print(lm_non_instruct.get_latest_response())

lm_instruct = InstructionModel(model, params_model, params_tokenizer, system_prompt)
lm_instruct.prompt("where is delhi?")
print(lm_instruct.get_latest_response())

print("Conversation History for Non Instruction Model:")
lm_non_instruct.print_conversation_history()

print("Conversation History for Instruction Model:")
lm_instruct.print_conversation_history()
```
