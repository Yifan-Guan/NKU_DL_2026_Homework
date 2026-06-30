import os

import torch

from dcgan import generate_images_from_noise, generate_latent_traversal, summarize_latent_traversal


def build_custom_noise(batch_size=8, nz=100, seed=42):
    torch.manual_seed(seed)
    return torch.randn(batch_size, nz)


def build_latent_base_noise(batch_size=8, nz=100, seed=123):
    torch.manual_seed(seed)
    return torch.randn(batch_size, nz)


if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)

    # 1) Use a custom batch of random noise to generate 8 images.
    custom_noise = build_custom_noise(batch_size=8, nz=100, seed=42)
    generate_images_from_noise(
        custom_noise,
        save_path="./results/dcgan_custom_8.png",
        nz=100,
    )

    # 2) Pick 5 random latent dimensions from 100 and traverse each one 3 times.
    torch.manual_seed(7)
    selected_indices = torch.randperm(100)[:5].tolist()
    base_noise = build_latent_base_noise(batch_size=8, nz=100, seed=123)
    traversal_result = generate_latent_traversal(
        base_noise=base_noise,
        selected_indices=selected_indices,
        step_values=[-2.0, 0.0, 2.0],
        save_path="./results/dcgan_latent_traversal_15x8.png",
        nz=100,
    )

    print("Custom 8-image sample saved to:", "./results/dcgan_custom_8.png")
    print("Latent traversal sample saved to:", traversal_result["save_path"])
    print("Selected latent indices:", traversal_result["selected_indices"])
    print("Traversal labels:")
    for line in summarize_latent_traversal(traversal_result["row_labels"]):
        print("  -", line)