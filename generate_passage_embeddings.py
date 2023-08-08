"""Encodes all passages into matrices, stores them on disk, and builds data 
structures for efficient search.

Sources: 
* https://github.com/stanford-futuredata/ColBERT#indexing
* https://github.com/stanford-futuredata/ColBERT/blob/d5b4a0453361ebcf09b9d62330ae471cf6cfe6dd/docs/intro.ipynb
* scripts/ColBERT/colbert/tests/index_updater_test.py
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", "%m/%d/%Y %H:%M:%S")
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def main():

    # args:
    parser = ArgumentParser()
    parser.add_argument("in_file", type=str, help="Path to input tsv file with chunks")
    parser.add_argument("out_dir", type=str, help="Path to output directory")
    parser.add_argument("--model_path", type=str, default="../../data/models/colbertv2.0",
                        help="Path to ColBERT checkpoint")
    parser.add_argument("--overwrite", action="store_true", 
                        help="Overwrite index if it already exists")

    args = parser.parse_args()

    logger.info(f"Running ColBERTv2 indexing...")

    with Run().context(RunConfig(nranks=1, root=args.out_dir, index_root=args.out_dir)):
        config = ColBERTConfig(
            doc_maxlen=1000, # Set to 1000 to avoid truncation TODO default is 220
            nbits=2, 
            # experiment=None,
        )
        indexer = Indexer(checkpoint=args.model_path, config=config)
        indexer.index(
            name="default", # NOTE name is mandatory
            collection=args.in_file,
            overwrite=args.overwrite,
        )

    # touch DONE file:
    Path(args.out_dir).parent.joinpath("DONE").touch()
    # Path(args.out_dir) / "DONE".touch()

    logger.info(f"DONE!")


if __name__=='__main__':
    main()
