
# Installation
`pip install -r requirements.txt`

# Usage
## Closed QA task only.
You can check our model card here: [`llm4fun/vietrag-7b-v1.0`](https://huggingface.co/llm4fun/vietrag-7b-v1.0)
```py
from transformers import GenerationConfig, TextStreamer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import torch

question = "<your-question>"
context = "<your-context>"
instruction = 'You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.'
input = f"Dựa vào một số ngữ cảnh được cho dưới đây, trả lời câu hỏi ở cuối.\n\n{context}\n\nQuestion: {question}"
prompt_template = (
    "### System:\n"
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
prompt = prompt_template.format(instruction=instruction, input=input, output='')

torch_dtype = torch.bfloat16
model_id = "llm4fun/vietrag-7b-v1.0"
device = "cuda"

tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(
    model_id,
    config=LlamaConfig.from_pretrained(model_id),
    torch_dtype=torch_dtype
)

model = model.eval().to(device)

def generate(prompt, max_new_tokens=1024):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.13,
            max_new_tokens=max_new_tokens,
            # temperature=0.2,
            # top_p=0.95,
            # top_k=20,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )

    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

output = generate(prompt)
```
To tweak the model's answering style, feel free to replace the `instruction` part of the prompt. I reccommend you select one of these following instructions, because they are used during training. 
```py
instructions = [
    'You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.',
    'You are an AI assistant. You will be given a task. You must generate a detailed and long answer.',
    'You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.',
    'You are an smart assistant. Provide a direct, short and exact answer to the following question from its provided context.'
]
```
## RAG
First you need to prepare the corpus. Run:
```
cd vietnamese-llamma-rag
mkdir data
mv data_raw10k.zip data
unzip data/data_raw10k.zip
```
To encode this whole corpus into embeddings, please run the notebook `corpus_embeddings.ipynb`. However, this is only necessary if you want to try out the `better_rag.ipynb` that's I'm going to introduce :).

You can start with `basic_rag.ipynb` to get some quick experiments on how the `llm4fun/vietrag-7b-v1.0` model work within in simple RAG pipeline. In this version, we use only a BM25 retriever to retrieve contexts. For those who are interested in our VHAC's solution, please check `better_rag.ipynb`. In this improved version, we incorporate BM25 and semantic search, we also implement some techniques to make the retrieved contexts "better" before feeding them to the LLM.

## Training Closed QA model
We will soon release our details on training datasets.

## Limitations
- If the retrieved contexts are correct, the model is likely to give an accurate answer. However, if wrong contexts are retrieved, the model will hallucinate almost everytime. I think this is some kind of "the model does not know that it does not know" behavior.
- There is this weird behavior. If the context fed to the model is not parargaphs with complete sentences, the model will have higher chance of generating repeating tokens.
