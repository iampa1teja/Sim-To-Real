from core.tasks.base import load_task_spec
from core.tasks.generic import GenericTask


TASK = GenericTask(load_task_spec('prim_regrasp_after_drop'))
