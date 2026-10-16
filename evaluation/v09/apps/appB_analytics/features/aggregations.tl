# Reusable structured predicates and per-record transforms (pure).
from tensorlang import log

high_value := {D}.amount > 200
grocery := {D}.category == "grocery"
log_amount := log({D}.amount + 1.0)
