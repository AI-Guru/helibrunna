# Add the parent directory to the path to be able to import the languagemodel module
import os
script_folder = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(os.path.join(script_folder, ".."))

# Regular imports.
from source.languagemodel import LanguageModel
import traceback

# Define the model ids.
model_ids = [
    "TristanBehrens/bach-garland-xlstm",
    "TristanBehrens/bach-garland-mamba",
    "TristanBehrens/bach-garland-pharia",
    "TristanBehrens/bach-garland-transformer",
]

# Test all models.
successful_models = []
statistics = {}
for model_id in model_ids:
    print(f"Testing model: {model_id}...")
    try:
        model = LanguageModel(model_id, config_overrides={"backend": "vanilla"})
        model.summary()

        # One round of warmup.
        _ = model.generate(
            prompt="GARLAND_START",
            temperature=0.5,
            max_length=128,
            end_tokens=[],
            forbidden_tokens=[],
            return_structured_output=True
        )

        # Generate a sequence.
        output_dict = model.generate(
            prompt="GARLAND_START",
            temperature=0.4,
            max_length=128,
            end_tokens=[],
            forbidden_tokens=[],
            return_structured_output=True
        )
        print(output_dict)

        # Update the statistics.
        statistics[model_id] = {
            "elapsed_time": output_dict["elapsed_time"],
            "tokens_per_second": output_dict["tokens_per_second"],
        }

        # Append the model_id to the successful models.
        successful_models.append(model_id)
    
    except Exception as e:
        print(f"Error: {e} for model_id: {model_id}")
        traceback.print_exc()
        continue

# Print the successful models.
print(f"Successful models: {successful_models}")

# Print the statistics.
for model_id in successful_models:
    print(f"Statistics for model: {model_id}")
    statistics_model = statistics[model_id]
    for key, value in statistics_model.items():
        print(f"{key}: {value}")