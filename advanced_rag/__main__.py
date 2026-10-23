"""Command line: ingest, ask, evaluate.

    python -m advanced_rag ingest all            # build every index (naive + advanced)
    python -m advanced_rag ask gep "What is the growth outlook for South Asia?"
    python -m advanced_rag ask gep "..." --naive --show
    python -m advanced_rag eval all --answers    # retrieval table + answer checks
    python -m advanced_rag serve --port 8001     # web page at http://localhost:8001
"""
from __future__ import annotations

import argparse
import logging
import time

from . import config, pipeline, route


def _docs(name: str) -> list[str]:
    return list(config.DOCUMENTS) if name == "all" else [name]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m advanced_rag")
    sub = parser.add_subparsers(dest="command", required=True)
    choices = [*config.DOCUMENTS, "all"]

    p = sub.add_parser("ingest", help="parse, chunk, embed and index")
    p.add_argument("doc", choices=choices)

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("doc", choices=list(config.DOCUMENTS))
    p.add_argument("question")
    p.add_argument("--naive", action="store_true", help="use the baseline pipeline")
    p.add_argument("--show", action="store_true", help="print the retrieved passages")

    p = sub.add_parser("eval", help="score retrieval (and answers) on the question sets")
    p.add_argument("doc", choices=choices)
    p.add_argument("--answers", action="store_true", help="also generate and check answers")
    p.add_argument("--pause", type=float, default=0.0,
                   help="seconds between LLM calls, for a rate-limited key")

    p = sub.add_parser("serve", help="run the local web page")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)

    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "ingest":
        for doc in _docs(args.doc):
            for use_naive in (True, False):
                pipe = pipeline.get(doc, use_naive)
                start = time.time()
                count = pipeline.ingest(pipe, doc)
                print(f"{pipe.collection:12} {count:5} chunks  {time.time() - start:5.1f}s")

    elif args.command == "ask":
        pipe = pipeline.get(args.doc, args.naive)
        result = pipeline.ask(pipe, args.question)
        if pipe.router:
            print(f"filter: {route.describe(result.where)}\n")
        print(result.answer)
        print("\nSources:")
        for hit in result.hits:
            print(f"  [{hit.rank}] p.{hit.page} {hit.kind:6} {hit.heading[:90]}")
            if args.show:
                print("      " + hit.context[:700].replace("\n", "\n      ") + "\n")

    elif args.command == "serve":
        import uvicorn
        uvicorn.run("advanced_rag.server:app", host=args.host, port=args.port)

    elif args.command == "eval":
        from .evaluate import run
        print(run(_docs(args.doc), args.answers, args.pause))


if __name__ == "__main__":
    main()
