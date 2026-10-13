"""持续推动计划的条件交接与原生进展；不选择侧面或原子动作。"""
from jev4mujoco.contracts.local_intent import PushExitProgressV1, PushGoalObservationV1, PushPreparationV1, LocalIntentCandidateV1
from jev4mujoco.runtime.push_interaction import (push_goal_geometry_v1, push_reposition_ready_v1,
    push_side_aligned_v1, current_push_plan_side_v1, push_interaction_mode_v1)


def native_push_goal_observation_v1(snapshot):
    geometry = push_goal_geometry_v1(snapshot)
    if geometry.get("status") != "observed" or snapshot.skill.goal.desired_value is not True:
        return None
    exits = tuple(PushExitProgressV1(snapshot.state_id, snapshot.action_epoch,
        snapshot.visual_scene.observation_id, geometry["exit_region_ref"],
        tuple(tuple(p) for p in geometry["exit_region_boundary_xy_m"]), geometry["required_clearance_m"],
        tuple(item["direction_xy"]), item["remaining_translation_m"])
        for item in geometry.get("directional_exit_requirements", ()))
    return PushGoalObservationV1(snapshot.state_id, snapshot.action_epoch,
        snapshot.visual_scene.observation_id, geometry["vertex_signed_boundary_distances_m"],
        geometry.get("vertex_outside_distances_m"), geometry.get("footprints_overlap"),
        geometry.get("footprint_clearance_m"), geometry["goal_satisfied"], exits)


def native_push_goal_improved_v1(previous, current, minimum_progress_m, *, direction_xy=None, best_exit_progress=()):
    """比较各关系自身的米制量，不合成跨任务评分，不假设物体跟随指令。"""
    if previous is None or current is None:
        return None
    if len(previous.vertex_signed_boundary_distances_m) != len(current.vertex_signed_boundary_distances_m):
        return None
    if current.goal_satisfied and not previous.goal_satisfied:
        return True
    if previous.vertex_outside_distances_m is not None and current.vertex_outside_distances_m is not None:
        return max(previous.vertex_outside_distances_m) - max(current.vertex_outside_distances_m) >= minimum_progress_m
    plane = next((p for p in current.exit_progress if p.direction_xy == direction_xy), None)
    old = None if plane is None else next((p for p in best_exit_progress if p.same_plane(plane)), None)
    # 旧记录或区域/方向无法对应时不沿用有平台期的逐顶点最小距离。
    return None if old is None else old.remaining_translation_m - plane.remaining_translation_m >= minimum_progress_m


def update_best_exit_progress_v1(best, current, minimum_progress_m):
    """各方向保留独立的真实最佳观测，不拼成一个虚构物体状态。"""
    result = list(best)
    for plane in current.exit_progress:
        index = next((i for i, old in enumerate(result) if old.same_plane(plane)), None)
        if index is None:
            result.append(plane)  # 新的可比较基线本身不是进展。
        elif result[index].remaining_translation_m - plane.remaining_translation_m >= minimum_progress_m:
            result[index] = plane
    return tuple(result)


def push_plan_step_v1(goal_execution, snapshot):
    """返回已经由模型选择的计划的当前义务；未知时保留计划并仅观察。"""
    from jev4mujoco.runtime.local_intent import local_intent_candidates_v1
    side = current_push_plan_side_v1(goal_execution.plan, snapshot)
    mode = push_interaction_mode_v1(snapshot)
    if native_push_goal_observation_v1(snapshot) is None and mode != "blocked":
        return goal_execution.stage, LocalIntentCandidateV1("observe_push_evidence",
            "Wait for current native goal geometry; historical progress cannot authorize motion.",
            ("hold",), "phase_contract", reference_ref=snapshot.skill.subject_ref), "unknown"
    if mode in ("unknown", "blocked"):
        candidates = local_intent_candidates_v1(snapshot)
        return goal_execution.stage, candidates[0] if candidates else None, mode
    if side is None:
        return goal_execution.stage, None, "side_correspondence_lost"
    stage = goal_execution.stage
    if stage == "advance" and mode != "advance":
        stage = "contact"
    if stage == "align_side" and push_reposition_ready_v1(snapshot) is not True:
        stage = "clearance"
    if stage == "clearance" and push_reposition_ready_v1(snapshot) is True:
        stage = "align_side"
    if stage == "align_side" and push_side_aligned_v1(snapshot, side):
        stage = "contact"
    if stage == "contact" and mode == "advance":
        stage = "advance"
    preparation = None if stage == "advance" else PushPreparationV1(goal_execution.plan, stage)
    candidates = local_intent_candidates_v1(snapshot, push_contact_plan=goal_execution.plan,
        push_preparation=preparation)
    expected = {"clearance": f"prepare_push_side_edge_{side.edge_index}",
        "align_side": f"align_reprepared_push_edge_{side.edge_index}",
        "contact": f"contact_reprepared_push_edge_{side.edge_index}",
        "advance": "advance_surface_push"}[stage]
    return stage, next((c for c in candidates if c.choice_id == expected), None), mode
