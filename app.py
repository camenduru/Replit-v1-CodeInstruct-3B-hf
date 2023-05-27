import os
import gradio as gr
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# REPO = "teknium/Replit-v1-CodeInstruct-3B-fp16"
REPO = "/content/model"

os.system(f"apt -y install -qq aria2")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/resolve/main/pytorch_model.bin -d /content/model -o pytorch_model.bin")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/resolve/main/spiece.model -d /content/model -o spiece.model")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/config.json -d /content/model -o config.json")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/generation_config.json -d /content/model -o generation_config.json")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/special_tokens_map.json -d /content/model -o special_tokens_map.json")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/tokenizer_config.json -d /content/model -o tokenizer_config.json")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/added_tokens.json -d /content/model -o added_tokens.json")

# os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/adapt_tokenizer.py -d /content/model -o adapt_tokenizer.py")
# os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/attention.py -d /content/model -o attention.py")
# os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/blocks.py -d /content/model -o blocks.py")
# os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/configuration_mpt.py -d /content/model -o configuration_mpt.py")
# os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/hf_prefixlm_converter.py -d /content/model -o hf_prefixlm_converter.py")
# os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/meta_init_context.py -d /content/model -o meta_init_context.py")
# os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/modeling_mpt.py -d /content/model -o modeling_mpt.py")
# os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/norm.py -d /content/model -o norm.py")
# os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B-fp16/raw/main/param_init_fns.py -d /content/model -o param_init_fns.py")

description = """# <h1 style="text-align: center; color: white;"><span style='color: #F26207;'> Code Generation by Instruction with Replit-v1-CodeInstruct-3B-fp16 </h1>
<span style="color: white; text-align: center;"> This model is trained on a large amount of code and fine tuned on code-instruct datasets. You can type an instruction in the ### Instruction: section and received code generation.</span>"""

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.to(device)

model.eval()

custom_css = """
.gradio-container {
    background-color: #0D1525; 
    color:white
}
#orange-button {
    background: #F26207 !important;
    color: white;
}
.cm-gutters{
    border: none !important;
}
"""

def post_processing(prompt, completion):
    return prompt + completion

def code_generation(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9, eos_token_id=tokenizer.eos_token_id):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, use_cache=True, temperature=temperature, top_p=top_p, eos_token_id=eos_token_id)
    completion = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return post_processing(prompt, completion)

demo = gr.Blocks(
    css=custom_css
)

with demo:
    gr.Markdown(value=description)
    with gr.Row():
        input_col , settings_col  = gr.Column(scale=6), gr.Column(scale=6), 
        with input_col:
            code = gr.Code(lines=28,label='Input', value="### Instruction:\n\n### Response:")
        with settings_col:
            with gr.Accordion("Generation Settings", open=True):
                max_new_tokens= gr.Slider(
                    minimum=8,
                    maximum=128,
                    step=1,
                    value=48,
                    label="Max Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.5,
                    step=0.1,
                    value=0.2,
                    label="Temperature",
                )

    with gr.Row():
        run = gr.Button(elem_id="orange-button", value="Generate Response")

    event = run.click(code_generation, [code, max_new_tokens, temperature], code, api_name="predict")

demo.queue(max_size=40).launch(share=True)