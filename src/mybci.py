import click
from loguru import logger
from constants import TaskType, TaskParadigm

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Main entry point"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())



@cli.command()
@click.option('--subject', '-s', type=int, required=True, 
              help='Subject number (1-109)')
@click.option('--task-type', '-t', type=click.Choice([task_type.value for task_type in TaskType]),
              required=True, help='Experiment type')
@click.option('--task-paradigm', '-p', type=click.Choice([task_paradigm.value for task_paradigm in TaskParadigm]),
              required=True, help='Experiment type')
@click.option('--mode', '-m', type=click.Choice(['train', 'predict']), 
              default='train', help='Mode: train or predict')
@click.option('--use-wavelets', '-w', is_flag=True, help='Use wavelet transform or not (BONUS)')
def run(subject, task_type, task_paradigm, mode, use_wavelets, verbose):
    pass

@cli.command()
def evaluate():
    pass

@cli.command()
def visualize():
    pass