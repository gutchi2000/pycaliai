# Current-race bodyweight forward source

This is a data-source feasibility collector, not a numbered model experiment.
It records JRA-VAN `0B11` (`WH`) snapshots available before post time and does
not read results, payouts, odds, predictions, or bets.

## Contract

- Poll each verified-calendar race from T-90 through T-5.
- Save every attempt as a new immutable JSON file, including `no_WH_record`.
- Deduplicate successful snapshots by the SHA-256 of the original CP932 bytes;
  a changed WH record creates another immutable snapshot.
- Preserve `000`/`999` and malformed fields as explicit statuses. Never impute
  them as zero.
- Keep the raw CP932 bytes (hex + hash) so parsing remains independently auditable.
- Do not connect this store to production features until a separate reviewed
  research specification passes coverage, timing, and time-safety gates.

The local store is `data/forward_bodyweight/` and must not be published.

## Live operation (2026-09-26)

Windows task `PyCaLiAI_BodyWeight` runs the 32-bit Python collector directly
(no PowerShell COM bridge) every five minutes from 08:15 for nine hours.
It no-ops with exit 0 when the verified calendar for that date is absent.
`MultipleInstancesPolicy=IgnoreNew`, `WakeToRun=true`, and the execution limit
is four minutes. Disable or unregister this one task to stop collection; stored
snapshots remain immutable.

The first real run at 11:41 JST saw five races in the T-90..T-5 window. Two WH
records were already available (29 horses total), while three attempts were
stored as `no_WH_record`. A scheduled-task run then completed with exit 0 and
added another five attempt records.

## Parser correction

The existing `jvlink_changes.parse_wh()` treated JV-Data byte offsets as Python
Unicode character offsets, so Japanese names shifted every weight field and
all WH rows were silently discarded. It now reconstructs the original one-byte
COM mapping and slices the CP932 bytes. The legacy internal JSON view is kept;
rich missingness/status fields are used only by this collector.
