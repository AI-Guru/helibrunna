import sys
sys.path.append(".")
from source.languagemodel import LanguageModel
import random

model_path_or_repo = "output/jsfakes_garland_xlstm/run_20240823-1527/"
model = LanguageModel(
    model_path_or_repo,
    config_overrides={"context_length": 4096}
)

# Make input of length 4096
min_id = 0
max_id = model.tokenizer.vocab_size - 1
sequence_length = 2048 - 1
inputs = [random.randint(min_id, max_id) for _ in range(sequence_length)]
input_string = model.tokenizer.decode(inputs)
print(f"Input: {input_string}")

# Generating.
result = model.generate(input_string, max_length=4096, return_structured_output=True)
#print(result)
print(len(result["output"].split()))