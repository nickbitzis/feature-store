PORT ?= 8858
CARDS ?= 50000
DAYS ?= 60

.PHONY: install test lint format build stream skew-gate run docker-build compose-up screenshot clean

install:
	uv sync --python 3.12

test:
	uv run --group dev pytest -q

lint:
	uvx ruff check .

format:
	uvx ruff format .
	uvx ruff check . --fix

build:  ## generate 2M events, build both stores, train, check skew, benchmark latency (about a minute)
	uv run featstore build --cards $(CARDS) --days $(DAYS)

stream:  ## replay the event log through the stream processor again
	uv run featstore stream

skew-gate:  ## what CI runs: clean with the bug flag off, detected with it on
	uv run featstore skew --expect-clean
	uv run featstore skew --bug --expect-detected

run:
	uv run featstore serve --port $(PORT) --bootstrap

docker-build:
	docker build -t feature-store .

compose-up:
	docker compose up --build

screenshot:  ## needs the server running on $(PORT) and Google Chrome
	uv run python scripts/screenshot.py --port $(PORT)

clean:
	rm -rf .pytest_cache .ruff_cache data
