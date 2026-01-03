.PHONY: install test lint fmt clean

install:
	uv sync --dev
	uv pip install maturin pytest
	uv run maturin develop --release

test:
	uv run pytest tests/ -v

lint:
	cargo fmt -- --check
	cargo clippy -- -D warnings

fmt:
	cargo fmt

clean:
	cargo clean
	rm -rf target/
	rm -rf .pytest_cache/
	rm -rf **/__pycache__/
