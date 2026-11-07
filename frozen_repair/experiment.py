import time
from .adapter import new_instance, instance_state, restore_instance, destroy, Session, routes, check
from .common import digest
from .rng import stream, seed, local
from .diagnostics import Diagnostic, natural_summary
from .counterfactual import complete_point


def run_instance(ident, x, elite, actor, metadata, ledger, task):
    started = time.perf_counter()
    instance = new_instance(x, metadata)
    instance.create_initial_solution()
    initial_cost = check(instance, metadata["round_distances"])
    initialization_seconds = time.perf_counter()-started
    checkpoints = {}
    checkpoints[0] = instance_state(instance)
    evolution = []
    for iteration in range(1, 21):
        parts = (ident, "evolution", iteration)
        def evolve():
            with stream(*parts):
                destroy(instance, metadata)
                session = Session(instance, actor, metadata)
                actions, cost = session.finish()
            return dict(state=instance_state(instance), cost=cost, action_count=len(actions))
        evolved = ledger.cached("evolution_full", task, [iteration], evolve)
        instance = restore_instance(evolved["state"])
        evolution.append(dict(iteration=iteration, seed=seed(*parts), cost=evolved["cost"], actions=evolved["action_count"]))
        if iteration in (5, 20):
            checkpoints[iteration] = instance_state(instance)
    states = []
    for depth in (0, 5, 20):
        complete = restore_instance(checkpoints[depth])
        before_cost = check(complete, metadata["round_distances"])
        previous = routes(complete)
        with stream(ident, depth, "diagnostic-destroy"):
            destroy(complete, metadata)
        residual = instance_state(complete)
        diagnostic = Diagnostic(elite, previous, x)
        trajectories, selected, eligible_count = [], None, 0
        for trajectory in range(1, 5):
            parts = (ident, depth, "natural", trajectory)
            def sample_natural():
                chosen, opportunities = None, 0
                selection_rng = local(ident, depth, "decision-point")
                with stream(*parts):
                    session = Session(restore_instance(residual), actor, metadata)
                    actions = []
                    while not session.done:
                        d = session.distribution()
                        d.update(diagnostic.qualify(d))
                        target = session.sample(d)
                        if trajectory == 1 and d["Vnew"]:
                            opportunities += 1
                            if selection_rng.randrange(opportunities) == 0:
                                snap = session.snapshot()
                                chosen = dict(snapshot=snap, snapshot_hash=digest(snap), decision=d,
                                              policy_target=target)
                        actions.append(session.apply(d, target))
                    cost = check(session.instance, metadata["round_distances"])
                return dict(trajectory=dict(trajectory_id=trajectory, seed=seed(*parts), cost=cost, feasible=True,
                                            actions=actions, routes=routes(session.instance)),
                            chosen=chosen, opportunities=opportunities)
            sampled = ledger.cached("natural_full", task, [depth, trajectory], sample_natural)
            trajectories.append(sampled["trajectory"])
            if trajectory == 1:
                selected, eligible_count = sampled["chosen"], sampled["opportunities"]
        diagnostic.annotate(trajectories, before_cost)
        if selected is not None:
            selected["eligible_count"] = eligible_count
        point = complete_point(selected, actor, metadata, x, ident, depth, before_cost, diagnostic, ledger, task)
        states.append(dict(depth=depth, before_cost=before_cost, complete_state=checkpoints[depth],
                           residual_state=residual, residual_hash=digest(residual),
                           destroy_seed=seed(ident, depth, "diagnostic-destroy"), trajectories=trajectories,
                           natural_summary=natural_summary(trajectories), point=point,
                           missing_opportunity=point is None))
    return dict(instance_id=ident, instance=x, elite_routes=elite, customer_ids=x["customer_ids"],
                initial_cost=initial_cost, initialization_seconds=initialization_seconds,
                extra_operations=dict(initial_solution_constructions=1, evolution_destroys=20, diagnostic_destroys=3,
                                      saved_complete_states=3, saved_residual_states=3),
                states=states, evolution=evolution,
                elapsed_seconds=time.perf_counter()-started)
