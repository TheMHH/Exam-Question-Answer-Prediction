import argparse
import os
from train import ExamTrainer
from test import ExamEvaluator
from inference import ExamInferencer

def main():
    parser = argparse.ArgumentParser(description="Exam Question Answering Model")
    parser.add_argument("--mode", choices=["train", "test", "infer"], required=True,
                      help="Operation mode: train, test, or infer")
    parser.add_argument("--model", choices=["gemma-it", "gemma-pt"], required=True,
                      help="Model type to use")
    parser.add_argument("--checkpoint", help="Path to a model checkpoint")
    parser.add_argument("--image", help="Path to an exam question image file")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for training/evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    if args.mode == "train":
        trainer = ExamTrainer(
            model_type=args.model,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            checkpoint_path=args.checkpoint
        )
        
        trainer.train()

    elif args.mode == "test":
        evaluator = ExamEvaluator(
            model_type=args.model,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint
        )
        
        evaluator.evaluate()

    elif args.mode == "infer":
        if not args.image:
            raise ValueError("--image argument is required for inference mode")
        
        inferencer = ExamInferencer(
            model_type=args.model,
        )
        
        result = inferencer.predict_answer(args.image)
        
        if result['error']:
            print(f"Error: {result['error']}")
        else:
            print("\nPrediction Results:")
            print(result['generated_text'])

if __name__ == "__main__":
    main() 