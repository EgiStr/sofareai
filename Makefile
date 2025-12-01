.PHONY: build up down logs clean test

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +

test:
	docker compose run --rm training python -m unittest discover tests
	docker compose run --rm serving python -m unittest discover tests
