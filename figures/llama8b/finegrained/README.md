# Llama-8B Fine-Grained Figures

Fine-grained figures are split by how much extra structure they add beyond the
main Llama-8B plots.

- `one_axis/`: variants of the main figures with exactly one additional axis,
  such as shot, intervention, checkpoint, or dataset.
- `multi_axis/`: broader exploratory summaries that combine multiple axes,
  such as strategy-by-dataset reductions, checkpoint trajectories, rank
  profiles, and fine-grained bias/reasoning tradeoffs.

Each group contains:

- `pdf/`: paper-ready figures.
- `csv/`: plotted data tables.
