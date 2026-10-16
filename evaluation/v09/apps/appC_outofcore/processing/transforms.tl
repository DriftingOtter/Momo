# Chunk-lowered elementwise transform (pure operator).
from tensorlang import sin

widen := sin({X}) + {X} ** 2
