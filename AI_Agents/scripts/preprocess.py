"""
Preprocessing v2 — Supporto Attack Pattern Detection.

V1: 34 features (compatibilità con modelli esistenti)
    5 player + 2 velocity + 5 terrain + 4 boss pos + 4 boss status
    + 4 intent one-hot + 10 hazards (2×5)

V2: 51 features (V1 + 17 pattern features)
    + 8 primary pattern one-hot + 3 temporal + 2 primary velocity
    + 2 secondary (pattern + active) + 2 combat info

Regola d'oro: Tutti i valori normalizzati in [-1, +1] o [0, 1].
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# V1 — 34 FEATURES
# ═══════════════════════════════════════════════════════════════


def preprocess_state_v1(state_dict: dict) -> np.ndarray:
    features = []

    # 1. Player Status (5)
    features.append(state_dict.get("playerHealth", 0) / 20.0)
    features.append(state_dict.get("playerSoul", 0) / 100.0)
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))

    # 2. Player Velocity (2)
    features.append(np.clip(state_dict.get("playerVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("playerVelocityY", 0.0) / 20.0, -1.0, 1.0))

    # 3. Terrain (5)
    terrain = state_dict.get("terrainInfo", [1.0] * 5)
    if not terrain or len(terrain) < 5:
        terrain = [1.0] * 5
    features.extend([np.clip(t, 0.0, 1.0) for t in terrain[:5]])

    # 4. Boss Position (4)
    features.append(np.clip(state_dict.get("bossRelativeX", 0.0), -1.0, 1.0))
    features.append(np.clip(state_dict.get("bossRelativeY", 0.0), -1.0, 1.0))
    features.append(np.clip(state_dict.get("distanceToBoss", 50.0) / 50.0, 0.0, 1.0))
    features.append(float(state_dict.get("isFacingBoss", False)))

    # 5. Boss Status & Kills (4)
    features.append(np.clip(state_dict.get("bossVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("bossVelocityY", 0.0) / 20.0, -1.0, 1.0))
    features.append(state_dict.get("bossHealth", 100.0) / 100.0)
    features.append(state_dict.get("mantisLordsKilled", 0) / 3.0)

    # 6. Boss Intent One-Hot (4)
    boss_action = state_dict.get("bossAction", 0)
    features.extend([1.0 if boss_action == i else 0.0 for i in range(4)])

    # 7. Hazards — 2 × 5 features (10)
    hazards = state_dict.get("nearbyHazards", [])

    def add_hazard(h):
        features.append(np.clip(h.get("relX", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("relY", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityX", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityY", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("distance", 15.0) / 15.0, 0.0, 1.0))

    if len(hazards) > 0:
        add_hazard(hazards[0])
    else:
        features.extend([0.0] * 5)
    if len(hazards) > 1:
        add_hazard(hazards[1])
    else:
        features.extend([0.0] * 5)

    return np.array(features, dtype=np.float32)  # → 34 features


# ═══════════════════════════════════════════════════════════════
# V2 — 51 FEATURES (V1 + 17 attack pattern features)
# ═══════════════════════════════════════════════════════════════


def preprocess_state_v2(state_dict: dict) -> np.ndarray:
    base = preprocess_state_v1(state_dict)
    pattern_features = []

    # Primary pattern one-hot (8)
    primary_pattern = state_dict.get("primaryMantisPattern", 0)
    pattern_features.extend([1.0 if primary_pattern == i else 0.0 for i in range(8)])

    # Stato temporale (3)
    pattern_features.append(float(state_dict.get("primaryMantisWindUp", False)))
    pattern_features.append(float(state_dict.get("primaryMantisActive", False)))
    pattern_features.append(float(state_dict.get("primaryMantisRecovering", False)))

    # Primary velocity (2)
    pattern_features.append(
        np.clip(state_dict.get("primaryMantisVelX", 0.0) / 25.0, -1.0, 1.0)
    )
    pattern_features.append(
        np.clip(state_dict.get("primaryMantisVelY", 0.0) / 25.0, -1.0, 1.0)
    )

    # Secondary mantis (2)
    sec_pattern = state_dict.get("secondaryMantisPattern", 0)
    pattern_features.append(sec_pattern / 7.0)
    pattern_features.append(float(state_dict.get("secondaryMantisActive", False)))

    # Combat info (2)
    pattern_features.append(state_dict.get("activeMantisCount", 0) / 3.0)
    pattern_features.append(float(state_dict.get("anyMantisAttacking", False)))

    return np.concatenate([base, np.array(pattern_features, dtype=np.float32)])  # → 51


# ═══════════════════════════════════════════════════════════════
# SELETTORI
# ═══════════════════════════════════════════════════════════════


def preprocess_state(state_dict: dict, version: int = 2) -> np.ndarray:
    return (
        preprocess_state_v1(state_dict)
        if version == 1
        else preprocess_state_v2(state_dict)
    )


def preprocess_state_compat(state_dict: dict, target_dim: int) -> np.ndarray:
    return (
        preprocess_state_v1(state_dict)
        if target_dim <= STATE_DIM_V1
        else preprocess_state_v2(state_dict)
    )


# ═══════════════════════════════════════════════════════════════
# PATTERN REWARD BONUS
# ═══════════════════════════════════════════════════════════════


def compute_pattern_reward_bonus(state_dict: dict, action: int) -> float:
    bonus = 0.0
    is_dodge = action in [4, 6]
    is_attack = action in [5, 7]
    is_move = action in [0, 1, 2, 3]

    wind_up = state_dict.get("primaryMantisWindUp", False)
    active = state_dict.get("primaryMantisActive", False)
    recovering = state_dict.get("primaryMantisRecovering", False)
    secondary_active = state_dict.get("secondaryMantisActive", False)

    if wind_up:
        if is_dodge or is_move:
            bonus += 0.02
        if is_attack:
            bonus -= 0.01

    if active and is_dodge:
        bonus += 0.05

    if recovering and is_attack:
        bonus += 0.08

    if secondary_active and is_dodge:
        bonus += 0.03

    return bonus


# ═══════════════════════════════════════════════════════════════
# DIMENSIONI — VERIFICATE: python -c "from preprocess import *; ..."
#
# V1: 5 + 2 + 5 + 4 + 4 + 4 + 10 = 34
# V2: 34 + 8 + 3 + 2 + 2 + 2     = 51
# ═══════════════════════════════════════════════════════════════

STATE_DIM_V1 = 34
STATE_DIM_V2 = 51


def get_state_dim(version: int = 2) -> int:
    return STATE_DIM_V1 if version == 1 else STATE_DIM_V2


def get_stacked_dim(version: int = 2, stack_size: int = 4) -> int:
    return get_state_dim(version) * stack_size
