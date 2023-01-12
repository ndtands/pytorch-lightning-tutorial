training:
	python main.py --run_name jd_extract \
				   --dataset_version version_2 \
				   --max_seq_length 128 \
				   --train_batch_size 32 \
				   --num_epochs 3 \
				   --learning_rate 2e-5 
mlflow:
	mlflow ui

killmlflow:
	sudo kill -9 $(lsof -t -i:5000)

server:
	uvicorn api:api --reload 