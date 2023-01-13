training:
	python main.py --run_name jd_extract \
				   --dataset_version version_2 \
				   --max_seq_length 152 \
				   --train_batch_size 16 \
				   --num_epochs 25 \
				   --learning_rate 2e-5 \
				   --precision 32
mlflow:
	mlflow ui

killmlflow:
	sudo kill -9 $(lsof -t -i:5000)

server:
	uvicorn api:api --reload 