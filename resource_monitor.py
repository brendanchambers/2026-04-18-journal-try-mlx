"""Resource monitoring and statistics tracking for MLX inference."""

import time
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


class ResourceMonitor:
    """Tracks memory usage and generation performance statistics."""

    def __init__(self):
        self.console = Console()
        self.process = psutil.Process()
        self.initial_memory = 0
        self.model_loaded_memory = 0
        self.final_memory = 0
        self.start_time = 0
        self.first_token_time = None
        self.end_time = 0
        self.token_count = 0

    def get_memory_mb(self):
        """Get current process memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def record_initial(self):
        """Record initial memory before model loading."""
        self.initial_memory = self.get_memory_mb()

    def record_model_loaded(self):
        """Record memory after model is loaded."""
        self.model_loaded_memory = self.get_memory_mb()

    def start_generation(self):
        """Mark the start of text generation."""
        self.start_time = time.time()

    def record_first_token(self):
        """Record the time when the first token is generated."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def record_token(self):
        """Increment token counter."""
        self.token_count += 1

    def end_generation(self):
        """Mark the end of text generation and record final memory."""
        self.end_time = time.time()
        self.final_memory = self.get_memory_mb()

    def display_statistics(self):
        """Display formatted performance statistics using rich."""
        total_time = self.end_time - self.start_time
        time_to_first_token = (
            self.first_token_time - self.start_time if self.first_token_time else 0
        )
        tokens_per_second = self.token_count / total_time if total_time > 0 else 0

        # Create memory usage table
        memory_table = Table(title="Memory Usage", show_header=True, header_style="bold cyan")
        memory_table.add_column("Stage", style="dim")
        memory_table.add_column("Memory (MB)", justify="right", style="green")

        memory_table.add_row("Initial", f"{self.initial_memory:.2f}")
        memory_table.add_row("Model Loaded", f"{self.model_loaded_memory:.2f}")
        memory_table.add_row("After Generation", f"{self.final_memory:.2f}")
        memory_table.add_row(
            "Model Memory Increase",
            f"{self.model_loaded_memory - self.initial_memory:.2f}",
            style="bold green",
        )

        # Create performance table
        perf_table = Table(title="Performance Metrics", show_header=True, header_style="bold cyan")
        perf_table.add_column("Metric", style="dim")
        perf_table.add_column("Value", justify="right", style="yellow")

        perf_table.add_row("Time to First Token", f"{time_to_first_token:.3f} seconds")
        perf_table.add_row("Total Generation Time", f"{total_time:.3f} seconds")
        perf_table.add_row("Tokens Generated", f"{self.token_count} tokens")
        perf_table.add_row(
            "Mean Tokens per Second",
            f"{tokens_per_second:.2f} tok/s",
            style="bold yellow",
        )

        # Display both tables
        self.console.print()
        self.console.print(memory_table)
        self.console.print()
        self.console.print(perf_table)
        self.console.print()
