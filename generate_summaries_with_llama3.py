from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,5,6,7'
import json

# Load the Llama-3 model using vLLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
xsum = load_dataset("xsum")
# Optional: Adjust Sampling Parameters
sampling_params = SamplingParams(
    max_tokens=150,  # Number of tokens for summary
    temperature=0.7, # Controls randomness
    top_p=0.9,       # Top-p sampling for diversity
)
xsum_first_50k = xsum["train"][:50000]
documents = [instance for instance in xsum_first_50k["document"]]
# template = f"""
#               Write a summary of the following text delimited by triple backticks.
#               Return your response which covers the key points of the text, in 200 words or less.
#               ```{text}```
#               SUMMARY:
#            """
formatted_inputs =  [f"""Write a summary of the following text delimited by triple backticks. Return your response which covers the key points of the text, in 200 words or less.```{doc}```SUMMARY:""" for doc in documents]
batch_size = 100
# Perform inference for summarization
new_data = []
# for index in (0, len(formatted_inputs), batch_size):
for index in (0, 200, batch_size):
    batch_outputs = llm.generate(formatted_inputs[index:index+batch_size], sampling_params=sampling_params)
    for inner_index_in_batch in range(batch_size):
        new_data.append({"source":documents[index+inner_index_in_batch], "summary":batch_outputs[inner_index_in_batch].outputs[0].text})

# output = llm.generate(formatted_inputs, sampling_params)
# todo check this
# summary = output[0]

# print("Generated Summary:", summary)
with open("llama3_xsum_summaries_50k.json","w") as f:
    json.dump(new_data, f)
