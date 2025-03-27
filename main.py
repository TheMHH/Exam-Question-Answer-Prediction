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
        # Initialize trainer
        trainer = ExamTrainer(
            model_type=args.model,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs
        )
        
        # Train the model
        trainer.train(checkpoint_path=args.checkpoint)

    elif args.mode == "test":
        # Initialize evaluator
        evaluator = ExamEvaluator(
            model_type=args.model,
            batch_size=args.batch_size
        )
        
        # Evaluate the model
        evaluator.evaluate(checkpoint_path=args.checkpoint)

    elif args.mode == "infer":
        if not args.image:
            raise ValueError("--image argument is required for inference mode")
        
        # Initialize inferencer
        inferencer = ExamInferencer(args.model)
        
        # Predict answer for the exam question
        result = inferencer.predict_answer(
            args.image,
            checkpoint_path=args.checkpoint
        )
        
        print("\nPrediction Results:")
        print(f"Predicted Answer: {result['predicted_answer']}")
        print("\nConfidence Scores:")
        for answer, score in result['confidence_scores'].items():
            print(f"{answer}: {score:.4f}")

if __name__ == "__main__":
    main() 