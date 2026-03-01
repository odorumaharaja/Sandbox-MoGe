"""
MoGe depth estimation application
Processing logic (inference.py) and UI definition (ui.py) are separated
"""

import click
from inference import MoGeInference
from ui import create_demo


@click.command(help='Web demo')
@click.option('--server', 'server', default='localhost', help='The IP address of the server to bind to.')
@click.option('--share', is_flag=True, help='Whether to run the app in shared mode.')
@click.option('--pretrained', 'pretrained_model_name_or_path', default=None, help='The name or path of the pre-trained model.')
@click.option('--version', 'model_version', default='v2', help='The version of the model.')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Whether to use fp16 inference.')
def main(server:str, share: bool, pretrained_model_name_or_path: str, model_version: str, use_fp16: bool):
    """Main entry point"""
    inference = MoGeInference(pretrained_model_name_or_path, model_version, use_fp16)
    demo = create_demo(inference, share)
    demo.launch(server_name=server, share=share)


if __name__ == '__main__':
    main()