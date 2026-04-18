import time
import psutil
from mlx_lm import load, stream_generate


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def main():
    print("Loading model (will download ~2GB on first run)...")
    initial_memory = get_memory_usage_mb()

    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    model_loaded_memory = get_memory_usage_mb()

    prompt = "What is MLX?"
    print(f"\nPrompt: {prompt}\n")

    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # Track generation statistics
    token_count = 0
    first_token_time = None
    start_time = time.time()

    print("Response: ", end="", flush=True)

    # Stream tokens and track timing
    for token in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=100):
        if token_count == 0:
            first_token_time = time.time()
        print(token.text, end="", flush=True)
        token_count += 1

    end_time = time.time()
    final_memory = get_memory_usage_mb()

    # Calculate statistics
    total_time = end_time - start_time
    time_to_first_token = first_token_time - start_time if first_token_time else 0
    tokens_per_second = token_count / total_time if total_time > 0 else 0

    # Display statistics
    print("\n")
    print("=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    print(f"Memory Usage (initial):        {initial_memory:>10.2f} MB")
    print(f"Memory Usage (model loaded):   {model_loaded_memory:>10.2f} MB")
    print(f"Memory Usage (after gen):      {final_memory:>10.2f} MB")
    print(f"Memory Increase (model):       {model_loaded_memory - initial_memory:>10.2f} MB")
    print("-" * 60)
    print(f"Time to First Token:           {time_to_first_token:>10.3f} seconds")
    print(f"Total Generation Time:         {total_time:>10.3f} seconds")
    print(f"Tokens Generated:              {token_count:>10} tokens")
    print(f"Mean Tokens per Second:        {tokens_per_second:>10.2f} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
