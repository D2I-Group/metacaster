
from . import (
    bash,
    edit_file,
    finalize_round,
    glob,
    grep,
    multi_edit,
    read_file,
    read_image,
    run_round_evaluation,
    write_file,
)

TOOL_CONTEXT: dict = {}

TOOLS = [
    read_file.SCHEMA,
    write_file.SCHEMA,
    edit_file.SCHEMA,
    multi_edit.SCHEMA,
    grep.SCHEMA,
    glob.SCHEMA,
    bash.SCHEMA,
    read_image.SCHEMA,
    run_round_evaluation.SCHEMA,
    finalize_round.SCHEMA,
]

TOOL_HANDLERS = {
    "read_file": read_file.handler,
    "write_file": write_file.handler,
    "edit_file": edit_file.handler,
    "multi_edit": multi_edit.handler,
    "grep": grep.handler,
    "glob": glob.handler,
    "bash": bash.handler,
    "read_image": read_image.handler,
    "run_round_evaluation": run_round_evaluation.handler,
    "finalize_round": finalize_round.handler,
}
