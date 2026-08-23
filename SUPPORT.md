# Support

ElideDB is an early release under active development. Here is how to get
help, and what to expect.

## Before filing

The [manual](docs/USAGE.md) covers install, ingest, query, evaluation on your
own data, and a [troubleshooting table](docs/USAGE.md#8-troubleshooting) of
the failures we have seen. Most first-run problems are there.

## Filing

- **Something broke or a result is clearly wrong**: open a
  [bug report](https://github.com/exploring-curiosity/ElideDB/issues/new?template=bug_report.yml).
  The template asks for the command, the full output, and your environment;
  with those three we can usually reproduce it the same day.
- **You evaluated it on your own footage**: open a
  [result report](https://github.com/exploring-curiosity/ElideDB/issues/new?template=evaluation_result.yml).
  Good or bad, numbers on corpora we have never seen are the most useful
  thing you can send. Bad ones especially.
- **Anything else**: a plain issue is fine.

## What to expect

- Issues are triaged within **two business days**. You will get either a
  fix, a reproduction request, or an honest "this is a known limit" with a
  pointer to the [roadmap](docs/ROADMAP.md).
- Bugs on the supported surface (`relmo.api`, `relmo.cli`, the write and
  read paths) are fixed first. Anything in `deprecated/`, `bench/`, `eval/`
  or `scripts/` is research history and is not supported.
- Star the repository to follow progress; it moves quickly and the README
  is kept current.

## Commercial licensing, pilots, and early access

ElideDB is source-available for evaluation, testing, and research under the
[PolyForm Noncommercial License](LICENSE). For commercial use, a pilot, or
early access to the hosted version, write to **sr7431@nyu.edu** with a few
lines about your footage and what you want to ask it.

## Security

If you believe you have found a security issue, email **sr7431@nyu.edu**
rather than opening a public issue.
