"""阶段条件的语义定义；描述结果和证据，不选择动作或接近路线。"""

from dataclasses import dataclass
from functools import lru_cache

from jev4mujoco.planning.capabilities import CAPABILITY_TEMPLATES_V1, CapabilityKind


@dataclass(frozen=True, slots=True)
class GoalDefinitionV1:
    relation_condition: str
    parameter_refs: tuple[str, ...] = ()


GOAL_DEFINITIONS_V1 = {
    "in_region": GoalDefinitionV1("Entire subject footprint is contained in the region boundary."),
    "clear_of_region": GoalDefinitionV1(
        "Subject footprint is disjoint from the protected boundary and their minimum clearance reaches the configured margin.",
        ("surface_push_clear_of_region_margin_m",)),
    "inside": GoalDefinitionV1(
        "Subject footprint is contained in the volume boundary and its bottom and top are within the volume Z bounds plus volume_height_tolerance_m; transport checks only horizontal containment.",
        ("visual_relation_config.volume_height_tolerance_m",)),
    "on_surface": GoalDefinitionV1(
        "Subject bottom matches reference top height and footprint overlap fraction reaches the configured minimum; transport checks only overlap.",
        ("visual_relation_config.support_height_tolerance_m", "visual_relation_config.support_minimum_overlap_fraction")),
    "left_of": GoalDefinitionV1(
        "Robot-base +Y is left: subject minimum Y >= reference maximum Y + directional_gap_m.",
        ("visual_relation_config.directional_gap_m",)),
    "right_of": GoalDefinitionV1(
        "Robot-base -Y is right: subject maximum Y <= reference minimum Y - directional_gap_m.",
        ("visual_relation_config.directional_gap_m",)),
    "in_front_of": GoalDefinitionV1(
        "Robot-base +X is in front: subject minimum X >= reference maximum X + directional_gap_m.",
        ("visual_relation_config.directional_gap_m",)),
    "near": GoalDefinitionV1(
        "Footprint minimum clearance <= near_maximum_clearance_m.",
        ("visual_relation_config.near_maximum_clearance_m",)),
    "separated": GoalDefinitionV1(
        "Footprint minimum clearance >= separated_minimum_clearance_m.",
        ("visual_relation_config.separated_minimum_clearance_m",)),
    "pressed": GoalDefinitionV1(
        "Owned press-point Z minus observed button top Z reaches press_minimum_travel_m plus press_visual_confirmation_margin_m.",
        ("press_minimum_travel_m", "press_visual_confirmation_margin_m")),
}


@dataclass(frozen=True, slots=True)
class PhaseConditionSpecV1:
    measurement: str
    operator: str
    target: str | float | bool
    tolerance_parameter: str | None = None


C = PhaseConditionSpecV1


@dataclass(frozen=True, slots=True)
class PhaseConstraintSpecV1:
    condition: PhaseConditionSpecV1 | None
    scope: str = "active_phase"


# 名称映射为明确谓词；None 表示尚无完整核验实现，不是缺少一次观测。
PHASE_CONSTRAINT_SPECS_V1 = {
    "subject_supported": PhaseConstraintSpecV1(C("subject_supported", "equals", True)),
    "candidate_grasp": PhaseConstraintSpecV1(C("grasp_candidate", "equals", True)),
    "subject_held": PhaseConstraintSpecV1(C("subject_held", "equals", True)),
    "subject_not_held": PhaseConstraintSpecV1(C("grasp_not_held", "equals", True)),
    "supported_push_contact": PhaseConstraintSpecV1(C("supported_push_contact", "equals", True)),
    "press_point_alignment": PhaseConstraintSpecV1(C("tcp_press_point_xy_error_m", "at_most", "press_point_xy_tolerance_m")),
    "placement_relation_until_release": PhaseConstraintSpecV1(
        C("placement_horizontal_relation_satisfied", "equals", True), "until_observed_release"),
    "avoid_other_objects": PhaseConstraintSpecV1(None),
}

# Measurement meanings belong to the common contract, not a scene or action plan.
PHASE_MEASUREMENT_DEFINITIONS_V1 = {
    "subject_held": "The bound subject is confirmed held; this is an estimated state with the attached evidence source.",
    "subject_supported": "The bound subject is supported by the reported support region.",
    "visual_follow_confirmed": "The observed subject followed the measured TCP displacement during the grasp check.",
    "grasp_not_held": "The grasp estimate is NOT_HELD, rather than candidate, held, or unknown.",
    "grasp_candidate": "A candidate or confirmed grasp exists; a candidate alone does not confirm holding.",
    "tool_subject_contact_clear": "The reported tool contact mode is CLEAR.",
    "supported_push_contact": "The reported contact mode is PUSH_CONTACT with this bound subject.",
    "final_goal_relation": "Observed truth value of the bound final goal predicate.",
    "capability_goal_satisfied": "The bound goal matches its desired truth value, including configured geometric margins.",
    "gripper_position_m": "Measured gripper joint opening, not the commanded opening.",
    "gripper_commanded_position_m": "Commanded gripper joint opening, not proof it was reached.",
    "tcp_to_subject_top_xy_m": "Subject top XY minus TCP XY in robot_base; the third component is unused zero.",
    "tcp_subject_xy_error_m": "Euclidean magnitude of subject top XY minus TCP XY.",
    "tcp_to_subject_grasp_pose_m": "Subject centroid XY minus TCP XY; Z is nearest grasp-height-band boundary minus TCP Z, or zero within the band.",
    "tcp_grasp_xy_error_m": "Euclidean magnitude of subject centroid XY minus TCP XY.",
    "tcp_grasp_height_band_error_m": "Absolute distance from TCP Z to the permitted grasp-height band; zero inside the band.",
    "grasp_probe_lifted": "The probe reached the required lift relative to its recorded start.",
    "grasp_probe_failed": "The grasp probe reported failure.",
    "subject_bottom_clearance_m": "Observed subject bottom Z minus the effective clearance reference: max(source support Z, destination support Z) for ON_SURFACE, max(source support Z, container opening top Z) for INSIDE, otherwise source support Z.",
    "required_bottom_clearance_m": "Required clearance above the effective reference: placement_obstacle_clearance_m for INSIDE, otherwise transport_minimum_bottom_clearance_m.",
    "subject_bottom_clearance_deficit_m": "max(0, required_bottom_clearance_m - subject_bottom_clearance_m).",
    "placement_horizontal_relation_satisfied": "The bound placement relation is satisfied in XY; this does not establish final Z or support.",
    "subject_bottom_to_target_support_clearance_m": "Observed subject bottom Z minus destination support Z; positive is above support, negative is below.",
    "tcp_outside_subject_footprint_clearance_m": "Signed minimum XY distance from TCP to subject footprint boundary: positive outside, negative inside, zero on the boundary. This is a distance, not a chosen approach side.",
    "tcp_to_push_height_z_m": "(source support Z + surface_push_tcp_height_above_surface_m) - measured TCP Z. Negative means TCP is above the required height; positive means below. This is a signed Z correction, not current height.",
    "tcp_to_press_point_xy_m": "Owned press-point XY minus measured TCP XY; third component is unused zero, not button travel.",
    "tcp_press_point_xy_error_m": "Euclidean magnitude of owned press-point XY minus measured TCP XY; not button travel.",
    "stable_observation_count": "Number of distinct consecutive observations passing this stable verification contract.",
    "required_stable_observations": "Configured number of consecutive passing observations needed for this contract.",
}

PHASE_CONDITION_SPECS_V1 = {
    "gripper_open_v1": (C("gripper_position_m", "within", "gripper_opening_m", "gripper_tolerance_m"),
        C("gripper_commanded_position_m", "within", "gripper_opening_m", "gripper_tolerance_m")),
    "tcp_subject_xy_aligned_v1": (C("tcp_subject_xy_error_m", "at_most", "grasp_xy_tolerance_m"),),
    "tcp_grasp_pose_aligned_v1": (C("tcp_grasp_xy_error_m", "at_most", "grasp_xy_tolerance_m"),
        C("tcp_grasp_height_band_error_m", "at_most", 0.0)),
    "grasp_candidate_v1": (C("grasp_candidate", "equals", True),),
    "grasp_probe_lifted_v1": (C("grasp_candidate", "equals", True),
        C("grasp_probe_lifted", "equals", True), C("grasp_probe_failed", "equals", False)),
    "grasp_confirmed_v1": (C("subject_held", "equals", True),),
    "transport_clearance_v1": (C("subject_held", "equals", True),
        C("subject_bottom_clearance_m", "at_least", "required_bottom_clearance_m")),
    "held_subject_above_region_v1": (C("subject_held", "equals", True),
        C("placement_horizontal_relation_satisfied", "equals", True)),
    "transport_confirmed_v1": (C("subject_held", "equals", True),
        C("placement_horizontal_relation_satisfied", "equals", True), C("visual_follow_confirmed", "equals", True)),
    "release_candidate_v1": (C("grasp_not_held", "equals", True),
        C("gripper_position_m", "within", "gripper_opening_m", "gripper_tolerance_m")),
    "place_goal_stable_v1": (C("capability_goal_satisfied", "equals", True),
        C("grasp_not_held", "equals", True), C("subject_supported", "equals", True),
        C("stable_observation_count", "at_least", "required_stable_observations")),
    "push_effector_ready_v1": (C("gripper_position_m", "within", "surface_push_closed_gripper_position_m", "surface_push_gripper_position_tolerance_m"),
        C("gripper_commanded_position_m", "within", "surface_push_closed_gripper_position_m", "surface_push_gripper_position_tolerance_m"), C("grasp_not_held", "equals", True)),
    "push_precontact_aligned_v1": (C("tcp_outside_subject_footprint_clearance_m", "greater_than", 0.0),
        C("tcp_outside_subject_footprint_clearance_m", "within", "surface_push_precontact_clearance_m", "surface_push_precontact_xy_tolerance_m"),
        C("tcp_to_push_height_z_m", "within", 0.0, "surface_push_tcp_height_tolerance_m"),
        C("tool_subject_contact_clear", "equals", True), C("subject_supported", "equals", True), C("grasp_not_held", "equals", True)),
    "push_contact_established_v1": (C("supported_push_contact", "equals", True),
        C("subject_supported", "equals", True), C("grasp_not_held", "equals", True),
        C("tcp_to_push_height_z_m", "within", 0.0, "surface_push_tcp_height_tolerance_m")),
    "surface_push_goal_reached_v1": (C("capability_goal_satisfied", "equals", True),
        C("supported_push_contact", "equals", True), C("subject_supported", "equals", True), C("grasp_not_held", "equals", True)),
    "push_effector_retracted_v1": (C("tool_subject_contact_clear", "equals", True),
        C("subject_supported", "equals", True), C("grasp_not_held", "equals", True)),
    "surface_push_goal_stable_v1": (C("capability_goal_satisfied", "equals", True),
        C("tool_subject_contact_clear", "equals", True), C("subject_supported", "equals", True), C("grasp_not_held", "equals", True),
        C("stable_observation_count", "at_least", "required_stable_observations")),
    "press_point_aligned_v1": (C("tcp_press_point_xy_error_m", "at_most", "press_point_xy_tolerance_m"),),
    "button_pressed_v1": (C("capability_goal_satisfied", "equals", True),
        C("supported_push_contact", "equals", True),
        C("tcp_press_point_xy_error_m", "at_most", "press_point_xy_tolerance_m")),
}


@dataclass(frozen=True, slots=True)
class PhaseDefinitionV1:
    objective: str
    actor: str
    active_axes: tuple[str, ...]
    maintain: tuple[str, ...]
    parameter_refs: tuple[str, ...]


# Keys are the same named contracts dispatched by RuntimeVerifierV1.
# Parameters are deployment configuration references, never slow-brain numbers.
PHASE_DEFINITIONS_V1 = {
    "gripper_open_v1": PhaseDefinitionV1(
        "Reach the configured open gripper position.", "gripper", (), (),
        ("gripper_opening_m", "gripper_tolerance_m")),
    "tcp_subject_xy_aligned_v1": PhaseDefinitionV1(
        "Align TCP above the observed subject in XY; approach without disturbing other objects.",
        "tcp", ("x", "y"), ("subject_supported", "avoid_other_objects"),
        ("grasp_xy_tolerance_m",)),
    "tcp_grasp_pose_aligned_v1": PhaseDefinitionV1(
        "Keep XY alignment and put TCP within the observed subject grasp-height band.",
        "tcp", ("x", "y", "z"), ("subject_supported", "avoid_other_objects"),
        ("grasp_xy_tolerance_m",)),
    "grasp_candidate_v1": PhaseDefinitionV1(
        "Close the gripper and obtain a grasp candidate; obstruction alone does not confirm holding.",
        "gripper", (), ("avoid_other_objects",), ()),
    "grasp_probe_lifted_v1": PhaseDefinitionV1(
        "Lift to test whether the subject follows TCP; hold to observe stability.",
        "tcp_and_subject", ("z",), ("candidate_grasp",), ()),
    "grasp_confirmed_v1": PhaseDefinitionV1(
        "Confirm stable visual following after the lift probe.",
        "subject", (), ("candidate_grasp",), ()),
    "transport_clearance_v1": PhaseDefinitionV1(
        "Raise the held subject bottom above the source and relevant destination obstacle height.",
        "tcp_and_subject", ("z",), ("subject_held", "avoid_other_objects"),
        ("transport_minimum_bottom_clearance_m", "placement_obstacle_clearance_m")),
    "held_subject_above_region_v1": PhaseDefinitionV1(
        "Move the held subject until the placement relation holds in the horizontal plane; a relation is not center coincidence.",
        "tcp_and_subject", ("x", "y"), ("subject_held", "avoid_other_objects"),
        ("visual_relation_config.directional_gap_m",
         "visual_relation_config.near_maximum_clearance_m",
         "visual_relation_config.separated_minimum_clearance_m",
         "visual_relation_config.support_minimum_overlap_fraction")),
    "transport_confirmed_v1": PhaseDefinitionV1(
        "Confirm the placement relation in the horizontal plane and visual following.",
        "subject", (), ("subject_held",), ()),
    "release_candidate_v1": PhaseDefinitionV1(
        "Adjust subject bottom clearance to the destination support and choose when to release.",
        "tcp_and_subject", ("z",), ("placement_relation_until_release",),
        ("gripper_opening_m", "gripper_tolerance_m")),
    "place_goal_stable_v1": PhaseDefinitionV1(
        "Confirm final goal, release, support and stability over distinct observations.",
        "subject", (), (), ("placement_required_consecutive_observations",
        "placement_minimum_observation_interval_s", "placement_maximum_centroid_shift_m")),
    "push_effector_ready_v1": PhaseDefinitionV1(
        "Close the tool to its configured position without holding an object.",
        "gripper", (), (), ("surface_push_closed_gripper_position_m",
        "surface_push_gripper_position_tolerance_m")),
    "push_precontact_aligned_v1": PhaseDefinitionV1(
        "Choose a feasible side from the final goal and observed geometry; approach the outside of the subject footprint with the configured clearance and tool height.",
        "tcp", ("x", "y", "z"), ("subject_supported", "avoid_other_objects"),
        ("surface_push_precontact_clearance_m", "surface_push_precontact_xy_tolerance_m",
         "surface_push_tcp_height_above_surface_m", "surface_push_tcp_height_tolerance_m")),
    "push_contact_established_v1": PhaseDefinitionV1(
        "Establish supported pushing contact at the configured tool height on your chosen side.",
        "tcp", ("x", "y", "z"), ("subject_supported", "subject_not_held", "avoid_other_objects"),
        ("surface_push_tcp_height_above_surface_m", "surface_push_tcp_height_tolerance_m")),
    "surface_push_goal_reached_v1": PhaseDefinitionV1(
        "Push the supported subject until its entire footprint satisfies the final goal; CLEAR_OF_REGION means leaving the protected area with clearance.",
        "tcp_and_subject", ("x", "y"), ("supported_push_contact", "subject_not_held", "avoid_other_objects"),
        ("surface_push_clear_of_region_margin_m",)),
    "push_effector_retracted_v1": PhaseDefinitionV1(
        "Separate the tool from the subject after pushing while preserving support and the achieved goal.",
        "tcp", ("x", "y", "z"), ("subject_supported", "subject_not_held"), ()),
    "surface_push_goal_stable_v1": PhaseDefinitionV1(
        "Confirm final goal, support and tool separation over distinct stable observations.",
        "subject", (), (), ("surface_push_clear_of_region_margin_m",
        "surface_push_required_consecutive_observations", "surface_push_minimum_observation_interval_s",
        "surface_push_maximum_centroid_shift_m")),
    "press_point_aligned_v1": PhaseDefinitionV1(
        "Align TCP with the observed owned press point in XY.",
        "tcp", ("x", "y"), ("avoid_other_objects",), ("press_point_xy_tolerance_m",)),
    "button_pressed_v1": PhaseDefinitionV1(
        "Maintain point alignment and press into the surface along its normal until visual travel confirms the button goal.",
        "tcp_and_button", ("x", "y", "z"), ("press_point_alignment",),
        ("press_point_xy_tolerance_m", "press_minimum_travel_m", "press_visual_confirmation_margin_m")),
}


@dataclass(frozen=True, slots=True)
class BoundSubtaskDefinitionV1:
    instance_id: str
    kind: str
    subject_ref: str | None
    source_ref: str | None
    target_ref: str | None
    goal: tuple[str, str, bool] | None
    phases: tuple[tuple[str, str, PhaseDefinitionV1], ...]


@lru_cache(maxsize=256)
def bound_subtask_definition_v1(
    instance_id: str, kind: str, subject_ref: str | None, source_ref: str | None,
    target_ref: str | None, goal: tuple[str, str, bool] | None,
) -> BoundSubtaskDefinitionV1:
    template = CAPABILITY_TEMPLATES_V1[CapabilityKind(kind)]
    return BoundSubtaskDefinitionV1(
        instance_id, kind, subject_ref, source_ref, target_ref, goal,
        tuple((phase.phase_id, phase.completion_contract,
               PHASE_DEFINITIONS_V1[phase.completion_contract]) for phase in template.phases),
    )
