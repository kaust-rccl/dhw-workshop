from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

model = AutoModel.from_pretrained("openai-community/gpt2-xl")
#("bigscience/T0_3B")

print(f'Zero 2 memory requirements\n {estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)}')
print(f'Zero 3 memory requirements\n {estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)}')