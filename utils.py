import os

def read_file(file_path: str):
    """Reads the content of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        raise FileNotFoundError(f"Error reading file {file_path}: {str(e)}")
