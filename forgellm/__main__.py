#!/usr/bin/env python3
"""
ForgeLLM CLI entry point
"""

import sys
from forgellm.cli.commands import setup_cli


def main():
    """Main entry point for the CLI."""
    return setup_cli()


if __name__ == "__main__":
    sys.exit(main()) 