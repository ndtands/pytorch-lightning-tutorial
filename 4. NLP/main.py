from train import Training
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)

    parser.add_argument('--model_name_or_path', type=str, default='xlm-roberta-base')
    parser.add_argument('--dataset_version', type=str, required=True)
    parser.add_argument('--label_all_tokens', type=bool, default=False)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=15)

    # model specific arguments
    parser.add_argument('--use_crf', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    
    args = parser.parse_args()
    Training(args)
    