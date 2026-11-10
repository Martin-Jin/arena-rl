"""Flatten and normalize structured simulation observations for neural networks."""

from collections.abc import Mapping, Sequence

from arena_ai.config import AttackType, Config, load_config
from arena_ai.sim.observation import Observation, fighter_observation
from arena_ai.sim.state import AGENT_ID, FighterState, sort_key

# The features of one fighter, in vector order. Add new features at the end, and bump
# ``observation.spec_version`` in configs/game.yaml: old checkpoints no longer fit.
FEATURE_NAMES = (
    "x",
    "y",
    "vx",
    "vy",
    "on_ground",
    "facing_dir",
    "health",
    "attack_windup_left",
    "attack_active_left",
    "attack_recovery_left",
    "parry_window_left",
    "parry_recovery_left",
    "parry_window",
    "counter_frames",
    "stagger_frames",
    "parry_held",
    "busy_frames_left",
    # One-hot of the attack in progress, all 0 when not attacking.
    *(f"attack_type_{attack_type.value}" for attack_type in AttackType),
    # Position relative to the acting fighter, so the network does not have to subtract
    # two absolute positions. All 0 in the acting fighter's own row.
    "rel_x_ahead",
    "rel_y",
    "rel_x_aimed",
)
# Horizontal distances are scaled by this many attack reaches (body width + attack range),
# so a swing that can reach the acting fighter sits near the middle of [-1, 1], not in a
# sliver around 0 as it would when divided by the arena width.
RELATIVE_X_REACHES = 3.0
# Vertical distances are scaled by this many fighter heights (a swing hits within one).
RELATIVE_Y_HEIGHTS = 2.0
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FEATURES_PER_FIGHTER = len(FEATURE_NAMES)


def observation_size(max_opponents: int) -> int:
    """Return the vector size: the acting fighter, ``max_opponents`` slots and the mask."""
    if max_opponents < 1:
        raise ValueError("max_opponents must be at least 1")
    return FEATURES_PER_FIGHTER * (max_opponents + 1) + max_opponents


def fighter_features(vector: Sequence[float], slot: int) -> dict[str, float]:
    """Return the named features of one row: slot 0 is the acting fighter, 1.. opponents."""
    start = slot * FEATURES_PER_FIGHTER
    row = vector[start : start + FEATURES_PER_FIGHTER]
    if len(row) != FEATURES_PER_FIGHTER:
        raise ValueError(f"observation has no fighter slot {slot}")
    return dict(zip(FEATURE_NAMES, row))


def slot_count(vector: Sequence[float]) -> int:
    """Return how many opponent slots a vector has (its ``max_opponents``)."""
    slots, remainder = divmod(len(vector) - FEATURES_PER_FIGHTER, FEATURES_PER_FIGHTER + 1)
    if slots < 1 or remainder:
        raise ValueError(f"{len(vector)} values is not a valid observation size")
    return slots


def opponent_mask(vector: Sequence[float], max_opponents: int) -> list[float]:
    """Return the mask: 1.0 for an opponent slot that holds a living opponent, else 0.0."""
    start = FEATURES_PER_FIGHTER * (max_opponents + 1)
    return list(vector[start : start + max_opponents])


# The config the scales were last built for. Building them is pure arithmetic on the
# config, so one entry is enough: a run uses a single config.
_scales_cache: tuple[Config, list[float]] | None = None


def _scales(config: Config) -> list[float]:
    """Return the divisor of each feature, in ``FEATURE_NAMES`` order.

    A feature that is already in [-1, 1] (flags, facing, the one-hot) has the divisor 1.0.
    Every divisor must be positive.
    """
    global _scales_cache
    if _scales_cache is not None and _scales_cache[0] is config:
        return _scales_cache[1]
    world = config.game.world
    fighter = config.game.fighter
    combat = config.game.combat
    reach = RELATIVE_X_REACHES * (fighter.width_m + combat.attack_range_m)
    by_name = {
        "x": world.arena_width_m,
        "y": config.game.observation.max_observed_height_m,
        "vx": fighter.move_speed_mps,
        "vy": fighter.jump_velocity_mps,
        "health": fighter.max_health,
        "attack_windup_left": combat.longest_windup_frames,
        "attack_active_left": combat.longest_active_frames,
        "attack_recovery_left": combat.longest_recovery_frames,
        "parry_window_left": combat.parry_window_frames,
        "parry_recovery_left": combat.parry_recovery_frames,
        "parry_window": combat.parry_window_frames,
        "counter_frames": combat.counter_window_frames,
        "stagger_frames": combat.parried_stagger_frames,
        "busy_frames_left": max(combat.longest_attack_frames, combat.parry_total_frames),
        "rel_x_ahead": reach,
        "rel_y": RELATIVE_Y_HEIGHTS * fighter.height_m,
        "rel_x_aimed": reach,
    }
    scales = [float(by_name.get(name, 1.0)) for name in FEATURE_NAMES]
    for name, scale in zip(FEATURE_NAMES, scales):
        if scale <= 0:
            raise ValueError(f"Normalization maximum for {name} must be positive, got {scale}")
    _scales_cache = (config, scales)
    return scales


def _fighter_row(
    fighter: Mapping[str, object],
    config: Config,
    acting: Mapping[str, object],
) -> list[float]:
    """Encode one fighter's state as ``FEATURES_PER_FIGHTER`` values clamped to [-1, 1].

    ``acting`` is the fighter the observation is for: the relative features are measured
    from it. The raw values are listed in ``FEATURE_NAMES`` order and divided by ``_scales``.
    """
    position = fighter["position"]
    velocity = fighter["velocity"]
    acting_position = acting["position"]
    x = float(position["x"])
    y = float(position["y"])
    dx = x - float(acting_position["x"])
    dy = y - float(acting_position["y"])
    attack_type = fighter["attack_type"]
    raw = [
        x,
        y - config.game.world.floor_y_m,
        float(velocity["x"]),
        float(velocity["y"]),
        float(bool(fighter["on_ground"])),
        float(fighter["facing_dir"]),
        float(fighter["health"]),
        float(fighter["attack_windup_left"]),
        float(fighter["attack_active_left"]),
        float(fighter["attack_recovery_left"]),
        float(fighter["parry_window_left"]),
        float(fighter["parry_recovery_left"]),
        float(fighter["parry_window"]),
        float(fighter["counter_frames"]),
        float(fighter["stagger_frames"]),
        float(bool(fighter["parry_held"])),
        float(fighter["busy_frames_left"]),
        *(float(attack_type == kind) for kind in AttackType),
        dx * float(acting["facing_dir"]),
        dy,
        # Positive when this fighter faces the acting fighter (its swing points at it).
        -dx * float(fighter["facing_dir"]),
    ]
    return [max(-1.0, min(1.0, value / scale)) for value, scale in zip(raw, _scales(config))]


def assemble_vector(
    agent_row: list[float],
    opponent_rows: Sequence[list[float] | None],
    max_opponents: int,
) -> list[float]:
    """Join rows into ``[acting fighter][opponent slot 0..K-1][mask 0..K-1]``.

    ``opponent_rows`` fill the slots from the first. ``None`` stands for a dead opponent:
    its slot is all zeros with mask 0.0, so the network can tell an absent opponent from
    one standing at x = 0. Unfilled slots are zeros with mask 0.0 as well.
    """
    if len(opponent_rows) > max_opponents:
        raise ValueError(
            f"{len(opponent_rows)} opponents do not fit observation.max_opponents={max_opponents}"
        )
    empty = [0.0] * FEATURES_PER_FIGHTER
    vector = list(agent_row)
    mask = []
    for row in opponent_rows:
        vector.extend(empty if row is None else row)
        mask.append(0.0 if row is None else 1.0)
    padding = max_opponents - len(opponent_rows)
    vector.extend(empty * padding)
    return vector + mask + [0.0] * padding


def fighter_rows(fighters: Mapping[str, FighterState], config: Config) -> dict[str, list[float]]:
    """Return the row of the agent and of every living fighter, measured from the agent.

    A row depends only on the fighter and the agent, not on which fighter's view it ends up
    in, so build the rows once per step and assemble every view from them.
    """
    combat = config.game.combat
    observations = {fid: fighter_observation(f, combat) for fid, f in fighters.items()}
    acting = observations[AGENT_ID]
    return {
        fid: _fighter_row(observation, config, acting)
        for fid, observation in observations.items()
        if fid == AGENT_ID or fighters[fid].health > 0
    }


def encode_observation(
    observation: Observation,
    config: Config | None = None,
) -> list[float]:
    """Return the vector ``[acting fighter][opponent slot 0..K-1][mask 0..K-1]``.

    ``K`` is ``observation.max_opponents``. Opponents are sorted by their id's natural
    order (``opponent_10`` after ``opponent_2``) and fill the slots from the first. A slot
    with no opponent, or a dead one, is all zeros with mask 0.0, so the network can tell
    an absent opponent from one standing at x = 0. Fewer opponents than slots is fine, so
    one network plays against any number up to ``K``.
    """
    if config is None:
        config = load_config()
    acting = observation["agent"]
    opponents = observation["opponents"]
    return assemble_vector(
        _fighter_row(acting, config, acting),
        [
            _fighter_row(opponents[fid], config, acting) if opponents[fid]["health"] > 0 else None
            for fid in sorted(opponents, key=sort_key)
        ],
        config.game.observation.max_opponents,
    )
