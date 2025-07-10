from detectron2.checkpoint import PeriodicCheckpointer
from typing import Any

class PeriodicCheckpointerOnlyOne(PeriodicCheckpointer):
    def step(self, iteration: int, **kwargs: Any) -> None:
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)
        if (iteration + 1) % self.period == 0:
            self.checkpointer.save("{}_recent".format(self.file_prefix), **additional_state)
        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)