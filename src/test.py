import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
def load_model():
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    print(model)

def magnitude_change():
    vector1 = torch.tensor([2.0, 0.0])
    vector2 = torch.tensor([0.0, 2.0])

    difference = vector2 - vector1
    magnitude_change = torch.norm(difference, p=2)
    print(f"Magnitude change: {magnitude_change.item()}")

    plt.figure(figsize=(5, 5))
    plt.quiver(0, 0, vector1[0], vector1[1], angles="xy", scale_units="xy", scale=1, color="tab:blue", label="vector1")
    plt.quiver(0, 0, vector2[0], vector2[1], angles="xy", scale_units="xy", scale=1, color="tab:orange", label="vector2")
    plt.plot([vector1[0], vector2[0]], [vector1[1], vector2[1]], "k--", label="difference")
    plt.scatter([vector1[0], vector2[0]], [vector1[1], vector2[1]], color=["tab:blue", "tab:orange"])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.title(f"Magnitude change: {magnitude_change.item():.4f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def length_change():
    vector1 = torch.tensor([2.0, 0.0])
    vector2 = torch.tensor([0.0, 2.0])

    length1 = torch.norm(vector1, p=2)
    length2 = torch.norm(vector2, p=2)
    length_change = length2 - length1
    print(f"Length change: {length_change.item()}")

if __name__ == "__main__":
    # load_model()

    magnitude_change()
    length_change()
