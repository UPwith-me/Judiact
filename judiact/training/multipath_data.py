"""Keep short-SFT field budgets while replacing only the action objective."""

from ..shared_audit import SharedAuditNode
from ..short_review import review_context
from .action_cost import TASKS, check_cost_population
from .multipath_objective import ALGORITHM, multipath_cost_table
from .short_data import ShortReviewData


def multipath_views(data, sid, outputs, epoch):
    views = []
    for index, output in enumerate(outputs):
        prior, observed = data.short_views(sid, review_context(output), epoch)[0]
        # Only categorical contexts depend on the sampled review. Identical
        # full/short references and localization retain one supervised forward.
        if index:
            observed = [e for e in observed if isinstance(e.node, SharedAuditNode)]
        views.append((prior, observed))
    return views


class MultipathReviewData(ShortReviewData):
    def __init__(self, release, bank, frozen_costs, *, reference_seed=42, stratified_exposure=False):
        super().__init__(release, bank, "natural", True, False,
                         stratified_exposure=stratified_exposure, reference_seed=reference_seed)
        # Accept the existing whole-population snapshot, but never its old
        # hard/gain soft-CE coefficients. This also checks root/class coverage.
        check_cost_population(frozen_costs, self)
        self.multipath_costs = frozen_costs
        self.multipath_tables = {task: multipath_cost_table(frozen_costs["tasks"][task]["confusion"])
                                 for task in TASKS}
        self.inventory["multipath"] = {
            "algorithm": ALGORITHM, "source_adapter": frozen_costs["source_adapter"],
            "tables": self.multipath_tables,
            "measure": "original task/root/member mass divided once by 4*m_y; equal task budgets",
            "process": "unchanged full/short 1:1 split, reliable stage fields, native and relation weights",
            "contexts": "fresh K independent student reviews per update; categorical budgets averaged over K; "
                "reference supervision once; relations use the existing log-mixture with uniform path weights",
        }
        context = self.inventory["multipath"]["contexts"]
        self.inventory["short_review"]["context"] = context
        self.inventory["shared_audit_sft"]["read_context"] = context
        self.inventory["shared_action_objective"].update(
            training_objective=ALGORITHM, normalization="frozen full-train mu/(4*m_y); exposure compensation once",
            inference="unchanged raw four-action argmax; natural is the inference transform, not the training loss")
