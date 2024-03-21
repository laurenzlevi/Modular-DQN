import os

def save_config(args: dict, relative_path: str):
    """Stores provided dictionary in a config.txt file under the provided relative path"""

    os.makedirs(relative_path, exist_ok=True)
    os.makedirs(relative_path + "/models/", exist_ok=True)

    file = open(f"{relative_path}/config.sh", 'w')

    argument_keys = args.keys()

    file_content = "python main.py " + ' '.join([f"--{key} {args[key]}" for key in argument_keys if args[key] is not None and args[key] is not False])
    file.write(file_content)

    file.close()