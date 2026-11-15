"""Command line: `stagetrain plan|run|status|report|kinds`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .core.config import TRACKS, ConfigError, load_config, parse_overrides
from .core.pipeline import CACHED, GateFailed, Pipeline, PipelineError
from .core.report import REPORT, write_report
from .core.stage import kinds


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in [header, *rows]]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _short_needs(needs: list[str], groups: dict[str, list[str]]) -> list[str]:
    """A stage that needs every run of a matrix is shown as needing the matrix: `sft (5 runs)`."""
    out, used = [], set()
    for group, runs in groups.items():
        if runs and set(runs) <= set(needs):
            out.append(f"{group} ({len(runs)} runs)")
            used.update(runs)
    return [n for n in needs if n not in used] + out


def _pipeline(args: argparse.Namespace) -> Pipeline:
    return Pipeline(load_config(args.config, parse_overrides(args.set or [])))


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def cmd_plan(args: argparse.Namespace) -> int:
    p = _pipeline(args)
    p.validate()
    selected = set(p.select(_split(args.only), args.start, args.until))
    rows = []
    for s in p.status():
        will = "run" if s.name in selected and (args.force or s.state != CACHED) else "skip"
        rows.append([s.name, s.kind, ", ".join(_short_needs(s.needs, p.config.groups)) or "-", s.state, will])
    print(f"{p.config.track} / {p.config.name}  ->  {p.run_dir}")
    print(_table(rows, ["stage", "kind", "needs", "state", "next run"]))
    return 0


def _refresh_report(p: Pipeline) -> None:
    """Rewrite `report.html`; a report that cannot be written never stops a run."""
    try:
        write_report(p)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("stagetrain").warning("could not write the report: %s", e)


def cmd_run(args: argparse.Namespace) -> int:
    p = _pipeline(args)
    refresh = None if args.dry_run or args.no_report else (lambda _manifest: _refresh_report(p))
    try:
        p.run(
            only=_split(args.only), start=args.start, until=args.until, force=args.force,
            dry_run=args.dry_run, allow_stale=args.allow_stale, after_stage=refresh,
        )
    except GateFailed as e:
        print(f"gate failed: {e}", file=sys.stderr)
        return 2
    finally:
        if refresh is not None:
            _refresh_report(p)
            logging.getLogger("stagetrain").info("report: %s", p.run_dir / REPORT)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = write_report(_pipeline(args), args.out)
    print(path)
    if args.open:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    p = _pipeline(args)
    if args.json:
        print(json.dumps([s.__dict__ for s in p.status()], indent=2))
        return 0
    rows = []
    for s in p.status():
        metrics = "  ".join(f"{k}={v:.4g}" for k, v in sorted(s.metrics.items())[:6])
        rows.append([s.name, s.state, metrics or "-", s.reason if s.state != CACHED else ""])
    print(_table(rows, ["stage", "state", "metrics", "note"]))
    return 0


def cmd_kinds(args: argparse.Namespace) -> int:
    for track in [args.track] if args.track else TRACKS:
        for kind, cls in kinds(track).items():
            doc = (cls.__doc__ or "").strip().splitlines()
            print(f"{kind:32s} {doc[0] if doc else ''}")
            if args.params:
                for name, f in cls.Params.model_fields.items():
                    if f.is_required():
                        default = "required"
                    elif f.default_factory is not None:
                        default = repr(f.default_factory())
                    else:
                        default = repr(f.default)
                    print(f"    {name:28s} {default[:120]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stagetrain", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser, selection: bool = True) -> None:
        sp.add_argument("config", help="pipeline YAML")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override: vars.x=1, seed=7, or <stage>.<param.path>=value")
        if selection:
            sp.add_argument("--only", help="comma-separated stage names")
            sp.add_argument("--from", dest="start", help="first stage to run")
            sp.add_argument("--until", help="last stage to run")
            sp.add_argument("--force", action="store_true", help="rerun finished stages")

    sp = sub.add_parser("plan", help="validate the pipeline and show what a run would do")
    common(sp)
    sp.set_defaults(fn=cmd_plan)

    sp = sub.add_parser("run", help="run the pipeline, resuming finished stages")
    common(sp)
    sp.add_argument("--dry-run", action="store_true", help="validate and list, run nothing")
    sp.add_argument("--allow-stale", action="store_true",
                    help="with --only/--from: accept an earlier stage whose configuration changed since it ran")
    sp.add_argument("--no-report", action="store_true", help="do not write report.html")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("report", help="write report.html: stages, gates, metrics and training curves")
    common(sp, selection=False)
    sp.add_argument("--out", help="where to write it (default: <run_dir>/report.html)")
    sp.add_argument("--open", action="store_true", help="open it in the browser")
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("status", help="show each stage's state and metrics")
    common(sp, selection=False)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("kinds", help="list the stage kinds of a track")
    sp.add_argument("track", nargs="?", choices=TRACKS)
    sp.add_argument("--params", action="store_true", help="also list each kind's parameters")
    sp.set_defaults(fn=cmd_kinds)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return args.fn(args)
    except (ConfigError, PipelineError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
