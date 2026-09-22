# Contributing to PyCaLiAI

Thanks for considering a contribution. PyCaLiAI welcomes focused improvements
to leak-safe evaluation, calibration, reproducible pipelines, tests,
documentation, and public-data adapters.

## Before opening a pull request

1. Open an issue first for a large behavioral change or new data dependency.
2. Keep third-party or licensed race data out of the pull request.
3. Do not include credentials, account identifiers, live betting instructions,
   or private operational files.
4. Preserve the time-based split and as-of semantics. A feature must use only
   information available at the prediction timestamp.
5. Add or update tests for changed behavior.

## Development setup

PyCaLiAI is developed on Python 3.11. Create a virtual environment, then run:

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

Some integrations require Windows-only software or separately licensed data.
Tests and documentation should degrade clearly when those optional inputs are
absent.

## Pull request checklist

- The change has a narrow purpose and a clear description.
- Tests pass for the affected area.
- No target leakage or train/validation/test overlap was introduced.
- Generated artifacts and large model files are excluded unless essential.
- Data provenance and licensing are documented for every new input.
- User-facing behavior and operational documentation are updated.

By contributing, you agree that your original contribution is licensed under
the repository's MIT License. You must have the right to submit everything in
your contribution.
