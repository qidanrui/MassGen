def update_summary(new_content):
    """
    Record your working process and final summary report, which can be shared with other agents.

    Args:
        new_content (str): The working summary to be added.

    Returns:
        None
    """
    file_path = "summary.txt"
    with open(file_path, "a") as f:
        f.write(new_content)


def check_updates():
    """
    Check other agents' current progress on the same task.

    Args:
        None

    Returns:
        str: The complete content of the summary.txt file as a string.
    """
    file_path = "summary.txt"
    with open(file_path) as f:
        return f.read()
