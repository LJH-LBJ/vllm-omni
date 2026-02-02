# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import logging
import sys
from pathlib import Path
from typing import Literal

from vllm.docs.mkdocs.hooks.generate_argparse import MarkdownFormatter, auto_mock

logger = logging.getLogger("mkdocs")

# Define root and doc output directories
ROOT_DIR = Path(__file__).parent.parent.parent.parent
ARGPARSE_DOC_DIR = ROOT_DIR / "docs/generated/argparse_omni"

# Ensure the repo root is in sys.path for dynamic imports
sys.path.insert(0, str(ROOT_DIR))

# Dynamically import or mock the OmniServeCommand subcommand
OmniServeCommand = auto_mock("vllm_omni.entrypoints.cli.serve", "OmniServeCommand")


# Function to create parser using subparser_init style CLI class
def create_parser_subparser_init(subcmd_class):
    """
    Create an argparse parser using subparser_init style CLI class, with MarkdownFormatter.
    """

    class DummySubparsers:
        def add_parser(self, name, **kwargs):
            import argparse

            return argparse.ArgumentParser(prog=name)

    dummy_subparsers = DummySubparsers()
    parser = subcmd_class().subparser_init(dummy_subparsers)
    parser.formatter_class = MarkdownFormatter
    return parser


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    """
    Entry point for doc generation. Builds doc directory and outputs markdown for each CLI command.
    """
    logger.info("Generating vllm-omni argparse documentation")
    logger.debug("Root directory: %s", ROOT_DIR.resolve())
    logger.debug("Output directory: %s", ARGPARSE_DOC_DIR.resolve())
    if not ARGPARSE_DOC_DIR.exists():
        ARGPARSE_DOC_DIR.mkdir(parents=True)

    # Register all CLI parsers; you can easily add more commands here
    parsers = {
        "omni_serve": create_parser_subparser_init(OmniServeCommand),
        # "another_cmd": create_parser_subparser_init(AnotherCommandClass),
    }

    for stem, parser in parsers.items():
        doc_path = ARGPARSE_DOC_DIR / f"{stem}.inc.md"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(super(type(parser), parser).format_help())
        logger.info("Argparse generated: %s", doc_path.relative_to(ROOT_DIR))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    on_startup("build", False)
