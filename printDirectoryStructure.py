import os

def print_directory_tree(startpath, prefix=""):
    """
    Recursively prints the directory tree starting from the given path.

    Args:
        startpath (str): The path to the starting directory.
        prefix (str): The prefix to use for the current level of the directory tree. Default is an empty string.
    """
    items = sorted(os.listdir(startpath))
    for i, item in enumerate(items):
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            is_last = (i == len(items) - 1)
            print(prefix + "├── " + item if not is_last else prefix + "└── " + item)
            new_prefix = prefix + ("│   " if not is_last else "    ")
            print_directory_tree(path, new_prefix)

# Example usage
if __name__ == "__main__":
    startpath = '/scratch/aniemcz/CAT'
    #print(startpath)
    print_directory_tree(startpath)
