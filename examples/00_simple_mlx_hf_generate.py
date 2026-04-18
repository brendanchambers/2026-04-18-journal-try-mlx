from mlx_lm import load, generate

model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"

def main():
    print("Loading model (will download ~2GB on first run)...")
    model, tokenizer = load(model_name)

    prompt = "What is MLX?"
    print(f"\nPrompt: {prompt}\n")

    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # Generate response
    output = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=100)
    print(f"Response: {output}\n")


if __name__ == "__main__":
    main()
