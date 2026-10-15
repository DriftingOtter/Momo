# Analysis-stage operators (sibling import; re-export for main).
#
# `pipelines/` sees only its siblings: cross-directory reuse flows
# through the application root (main_native.tl), which fans in
# operators/, inputs/, and pipelines/.
from preprocess import rescale

# Runtime alias re-exporting the imported operator for consumers.
method = rescale
