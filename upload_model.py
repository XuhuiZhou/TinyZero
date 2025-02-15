from huggingface_hub import HfApi, login
import os
import argparse

def upload_to_huggingface(
    model_path: str,
    repo_name: str,
    token: str = None,
    repo_type: str = "model",
    private: bool = False
):
    """
    Upload a model to Hugging Face Hub
    
    Args:
        model_path (str): Local path to the model files
        repo_name (str): Name for the repository on HuggingFace (format: 'username/model-name')
        token (str, optional): HuggingFace authentication token
        repo_type (str, optional): Type of repository ('model', 'dataset', or 'space')
        private (bool, optional): Whether to create a private repository
    """
    # Verify model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist")
        return

    # Login to Hugging Face
    if token:
        login(token=token)
    else:
        # Will look for token in ~/.huggingface/token
        login()
    
    # Initialize the HF API
    api = HfApi()
    
    # Create the repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_name,
            repo_type=repo_type,
            private=private,
            exist_ok=True
        )
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload the model files
    try:
        # Upload all files in the directory
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type=repo_type
        )
        print(f"Successfully uploaded model to {repo_name}")
        print(f"View your model at: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Error uploading files: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Hub")
    parser.add_argument("--token", required=True, help="HuggingFace authentication token")
    parser.add_argument("--model_path", default="./models/", required=True, help="Path to the model files")
    parser.add_argument("--repo_name", default="countdown-qwen2.5-3b", required=True, help="Name for the repository (username/model-name)")
    parser.add_argument("--repo_type", default="model", choices=["model", "dataset", "space"], 
                        help="Type of repository")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    
    args = parser.parse_args()
    
    upload_to_huggingface(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        repo_type=args.repo_type,
        private=args.private
    ) 