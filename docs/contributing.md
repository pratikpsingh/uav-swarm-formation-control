# Development workflow

## Before changing code

1. Write down the behavior and completion test before implementing it.
2. Keep the change inside the current development stage's scope.

## Local checks

```bash
uv sync
uv run ruff format .
uv run ruff check --fix .
uv run pyright
uv run pytest --cov
```

Install the optional local Git hooks with:

```bash
uv run pre-commit install
```

## Commit policy

Each commit should represent one understandable change and leave the checks passing. Use an
imperative Conventional Commit-style subject, for example:

```text
feat(formations): add normalized formation error
test(observations): cover missing-neighbor masks
docs(learning): explain generalized advantage estimation
```

Generated data, checkpoints, and local tracking directories must not be committed. Small fixtures,
resolved experiment configurations, and summarized results may be committed when needed to reproduce
a claim.
