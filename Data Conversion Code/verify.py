import torch
import math

def check_model_prediction(model_path, input_values):
    model = torch.jit.load("/Users/sidkrishna/Documents/College/Research/MuscleStiffness/FinePose-main/Matlab Conversion/universal_model.pt")
    model.eval()

    x = torch.tensor(input_values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        raw_output = model(x).item()  # ~0..1

    # If you want to treat raw_output as the “stiffness probability” directly:
    stiffness_probability = raw_output

    # Or if you want “bigger raw => smaller risk,” invert it:
    # stiffness_probability = 1.0 - raw_output

    print(f"Input: {input_values}")
    print(f'Raw output:         {raw_output:.4f}')
    print(f'Stiffness prob:     {stiffness_probability:.2%}')
    print()

if __name__ == "__main__":
    model_file = "universal_model.pt"
    low_input =  [0.10, 0.30, 0.10, 0.50, 0.30, 0.30, 0.30, 0.20, 0.20, 0.20]
    med_input =  [0.50, 0.40, 0.30, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    high_input = [0.80, 0.70, 0.70, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80]

    check_model_prediction(model_file, low_input)
    check_model_prediction(model_file, med_input)
    check_model_prediction(model_file, high_input)
