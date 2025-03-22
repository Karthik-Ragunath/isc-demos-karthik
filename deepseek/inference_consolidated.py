### INFO: This is a helper script to allow participants to confirm their model is working!
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig, PeftConfig, PeftModel

from fsdp_utils import AppState

adapter_name = "ExampleLora"

# INFO: This is a helper to map model names to StrongCompute Dataset ID's which store their weights!
model_weight_ids = {
    "DeepSeek-R1-Distill-Llama-70B": "e4b2dc79-79af-4a80-be71-c509469449b4",
    "DeepSeek-R1-Distill-Llama-8B": "38b32289-7d34-4c72-9546-9d480f676840",
    "DeepSeek-R1-Distill-Qwen-1.5B": "6c796efa-7063-4a74-99b8-aab1c728ad98",
    "DeepSeek-R1-Distill-Qwen-14B": "39387beb-9824-4629-b19b-8f7b8f127150",
    "DeepSeek-R1-Distill-Qwen-32B": "84c2b2cb-95b4-4ce6-a2d4-6f210afad36b",
    "DeepSeek-R1-Distill-Qwen-7B": "6e226f91-6b7d-46ff-9f1e-4740efaf9b0e",
}

# TODO: set this to the model you chose from the dropdown at container startup!
MODEL_NAME_SETME = "DeepSeek-R1-Distill-Qwen-7B"
mounted_dataset_path = f"/data/{model_weight_ids[MODEL_NAME_SETME]}"

# INFO: Loads the model WEIGHTS (assuming you've mounted it to your container!)
tokenizer = AutoTokenizer.from_pretrained(mounted_dataset_path)
model = AutoModelForCausalLM.from_pretrained(
    mounted_dataset_path, 
    use_cache=False, 
    torch_dtype=torch.bfloat16
)

# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0, # set to zero to see identical loss on all ranks
# )
# model = LoraModel(model, lora_config, adapter_name).to("cuda")

# Load consolidated adapter weights
adapter_path = "/shared/artifacts/18142399-96ff-4846-b55b-3be3822720f6/checkpoints/AtomicDirectory_checkpoint_56_consolidated_lora"
config = PeftConfig.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(model, adapter_path).to("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
state_dict = {"app": AppState(model, optimizer)}
dcp.load(state_dict=state_dict, checkpoint_id="/shared/artifacts/18142399-96ff-4846-b55b-3be3822720f6/checkpoints/AtomicDirectory_checkpoint_56_backup") ## UPDATE WITH PATH TO CHECKPOINT DIRECTORY

prompt = "What is the coefficient of $x^2y^6$ in the expansion of $\left(\frac{3}{5}x-\frac{y}{2}\right)^8$? Express your answer as a common fraction."

# https://arxiv.org/abs/2501.12948
deepseek_r1_input = f'''
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: {prompt}. Assistant:'''

encoding = tokenizer(deepseek_r1_input, return_tensors="pt")

input_ids = encoding['input_ids'].to("cuda")
attention_mask = encoding['attention_mask'].to("cuda")

generate_ids = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=500, do_sample=True, temperature=0.8)
answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(answer[0])
