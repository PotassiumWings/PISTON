import sys

import pydantic_argparse

if __name__ == '__main__':
    from configs.arguments import TrainingArguments
    from main import main


    def parse_args(arguments):
        parser = pydantic_argparse.ArgumentParser(
            model=arguments,
            prog="python app.py",
            description="Training model job.",
            version="0.0.1",
            epilog="Training model job.",
        )

        return parser.parse_typed_args()


    args = parse_args(TrainingArguments)
    sys.exit(main(args))
