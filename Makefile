up:
	docker-compose up -d

build:
	docker-compose up -d --build

exec-in-container = docker-compose exec app

shell:
	${exec-in-container} bash

down:
	docker-compose down

# Deletes data from milvus
nuke:
	docker-compose down
	rm -rf volumes

run: up
	${exec-in-container} streamlit run bot/app.py

populate-db:
	${exec-in-container} python bot/populate_db.py

run-llama:
	python -m llama_cpp.server --model models/7B/llama-2-7b.Q4_K_M.gguf  --n_gpu_layers 1

copy-req:
	docker cp test-app-1:app/requirements.txt ./requirements.txt




