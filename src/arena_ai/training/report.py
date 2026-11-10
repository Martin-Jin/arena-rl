"""Format the training statistics for printing."""

from arena_ai.training.train import TrainingState


def format_stats(state: TrainingState) -> str:
    """Return a multi-line summary of the latest iteration.

    Rolling figures cover the last ``stats_window`` episodes, which is much less
    noisy than the most recent episode. Missing values print as zero, so this is
    safe before the first episode finishes.
    """
    m, e = state.metrics, state.episode_stats

    def metric(name: str) -> float:
        return m.get(name, 0.0)

    def episode(name: str) -> float:
        return e.get(name, 0.0)

    # What the rollout spent outside the environment and the network: building the
    # batch and recording episodes.
    other = max(
        0.0,
        metric("rollout_seconds")
        - metric("env_seconds")
        - metric("act_seconds")
        - metric("value_seconds"),
    )

    lines = [
        f"[iter {state.iteration}] steps={state.environment_steps:,} "
        f"episodes={int(metric('episodes'))}",
        f"  results  win {episode('win_rate'):.0%}  "
        f"return {episode('mean_return'):+.2f}  "
        f"reward/step {metric('reward_per_step'):+.4f}  "
        f"(last {int(episode('episodes'))} episodes)",
        f"  policy   kl {metric('approx_kl'):.4f}  "
        f"clip {metric('clip_fraction'):.0%}  "
        f"explained var {metric('explained_variance'):.2f}  "
        f"entropy {metric('entropy'):.2f} (move {metric('entropy_move'):.2f} "
        f"jump {metric('entropy_jump'):.2f} attack {metric('entropy_attack'):.2f} "
        f"parry {metric('entropy_parry'):.2f} dodge {metric('entropy_dodge'):.2f})",
        f"  loss     total {metric('loss'):.4f}  "
        f"policy {metric('policy_loss'):.4f}  value {metric('value_loss'):.4f}",
        f"  actions  attack {metric('attack_rate'):.0%}  "
        f"parry {metric('parry_rate'):.0%}  dodge {metric('dodge_rate'):.0%}  "
        f"jump {metric('jump_rate'):.0%}  "
        f"left {metric('left_rate'):.0%}  "
        f"right {metric('right_rate'):.0%}  idle {metric('idle_rate'):.0%}",
        f"  combat   accuracy {episode('attack_accuracy'):.0%}  "
        f"parries {episode('parries'):.1f}/ep ({episode('parry_accuracy'):.0%})  "
        f"dealt {episode('damage_dealt'):.0f}  taken {episode('damage_taken'):.0f}  "
        f"near wall {episode('wall_fraction'):.0%}  "
        f"win length {episode('win_length'):.0f}  "
        f"loss length {episode('loss_length'):.0f}",
        f"  time     {metric('iteration_seconds'):.2f}s/iter  "
        f"rollout {metric('rollout_seconds'):.2f}s "
        f"(env {metric('env_seconds'):.2f}s  act {metric('act_seconds'):.2f}s  "
        f"value {metric('value_seconds'):.2f}s  other {other:.2f}s)  "
        f"update {metric('update_seconds'):.2f}s  "
        f"{metric('steps_per_second'):,.0f} steps/s",
    ]
    width = max([9, *map(len, state.opponent_stats)])  # Matchups like "approach+idle+runner".
    for name, stats in state.opponent_stats.items():
        lines.append(
            f"  vs {name:<{width}} win {stats['win_rate']:.0%}  "
            f"timeout {stats['truncation_rate']:.0%} "
            f"(ahead {stats['timeout_win_rate']:.0%})  "
            f"length {stats['mean_length']:.0f}  "
            f"({int(stats['episodes'])} episodes)"
        )
    return "\n".join(lines)
