"""Minimal MLX language model example with performance monitoring."""

from mlx_lm import load, stream_generate
from rich.console import Console
from rich.panel import Panel

from resource_monitor import ResourceMonitor

# ============================================================================
# Configuration Variables
# ============================================================================
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
PROMPT = "What is MLX?"
MAX_TOKENS = 512

# ============================================================================
# Main Program
# ============================================================================


def main():
    console = Console()
    monitor = ResourceMonitor()

    # Initialize monitoring
    monitor.record_initial()

    # Load model
    console.print(
        Panel.fit(
            f"Loading model: [cyan]{MODEL_NAME}[/cyan]\n"
            "(will download ~2GB on first run)",
            title="MLX Demo",
            border_style="blue",
        )
    )

    model, tokenizer = load(MODEL_NAME)
    monitor.record_model_loaded()

    # Display prompt
    console.print()
    console.print(f"[bold blue]Prompt:[/bold blue] {PROMPT}")
    console.print()

    # Format as chat message
    messages = [{"role": "user", "content": PROMPT}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    # Generate response with streaming
    console.print("[bold green]Response:[/bold green] ", end="")
    monitor.start_generation()

    for token in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=MAX_TOKENS):
        if monitor.token_count == 0:
            monitor.record_first_token()
        console.print(token.text, end="", highlight=False)
        monitor.record_token()

    monitor.end_generation()
    console.print("\n")

    # Display statistics
    monitor.display_statistics()


if __name__ == "__main__":
    main()

