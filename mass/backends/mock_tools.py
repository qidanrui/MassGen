def update_summary(new_content: str):
    """
    Record your working process and final summary report, which can be shared with other agents.

    Args:
        new_content (str): The working summary to be added.

    Returns:
        None
    """
    print(f"[UPDATE_SUMMARY] {new_content}")
    # In a real implementation, this would write to a shared file or database
    return "Summary updated successfully"

def check_updates():
    """
    Check other agents' current progress on the same task.

    Args:
        None

    Returns:
        str: The complete content of the summary.txt file as a string.
    """
    return "Agent 1 is working on the task. He has reviewed your task and believe your solution is correct."

def vote(agent_id: int):
    """
    Vote for the representative agent to solve the task. You can also vote for yourself.

    Args:
        agent_id (int): The ID of the agent to vote for.

    Returns:
        str: The result of the vote.
    """
    return f"Vote for agent {agent_id}"