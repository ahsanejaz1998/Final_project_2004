# Read the evaluation result
with open("evaluation_result.txt", "r") as file:
    result = file.read()
    accuracy = float(result.split(":")[1])

# Set a threshold for the accuracy
threshold = 0.93
if accuracy < threshold:
    raise ValueError(f"Model accuracy {accuracy:.2f} is below the threshold {threshold}")
