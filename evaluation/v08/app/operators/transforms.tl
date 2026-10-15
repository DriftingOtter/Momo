# Reusable numerical transforms (deliberately overlapping param names
# with calibration.tl to test cross-file scope behavior).
zscore := ({X} - {M}) / {S}
center := {X} - {M}
