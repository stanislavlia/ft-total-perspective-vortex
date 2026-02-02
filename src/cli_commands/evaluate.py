import click


def run_evaluate():
    """
    Evaluate pipeline across all subjects and experiments.

    Runs the full evaluation matrix:
    - All subjects (S001-S109)
    - All task types (motor_execution, motor_imagery)
    - All paradigms (left_right_hand, hands_feet)

    Outputs summary statistics and per-subject results.
    """
    # TODO: Implement evaluate
    # 1. Iterate over all subjects
    # 2. For each experiment configuration (task_type x paradigm)
    # 3. Run training and collect results
    # 4. Aggregate and display summary statistics
    click.echo("Evaluate not yet implemented")
