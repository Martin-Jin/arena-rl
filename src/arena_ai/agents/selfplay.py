"""The learner's own network as a training opponent (``agent`` in ``training.opponents``).

**Why:** a scripted opponent is a fixed target, and the learner can only get as good as the
best script. A copy of itself is always as strong as the learner and gets stronger with it.
**What:** ``SelfPlayPolicy`` plays an opponent with a copy of the learner's weights. It sees
the game from its own side, as the viewer's trained opponent does (``own_observation``), and
samples its moves. Only the learner's own transitions are trained on: the opponent's moves
are never stored or learned from. **How the weights arrive:** training sends a flat vector of
the learner's weights (``weights_of``) after every PPO update, and ``load_weights`` loads it
into one network that every policy in the process shares. The environments run in worker
processes, so each worker has its own copy, loaded by ``AsyncVectorEnv.set_opponent_weights``.
**Limits:**
- The opponent is not mirrored: it sees its own absolute position and facing, which differ
  from what the learner usually sees on the left side of the arena.
- Allies are not visible to it, so several ``agent`` opponents do not coordinate.
- It acts on the weights of the latest update, so it lags the learner by at most one update.
- It needs weights before its first move; stepping an environment without them raises.
Torch is imported only when a self-play opponent is used, so scripted training workers
never load it.
"""

from collections.abc import Mapping

from arena_ai.agents.scripted.episode import EpisodePolicy
from arena_ai.agents.scripted.registry import SELF_PLAY
from arena_ai.config import Config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.state import FighterState, enemies_of

# The network shared by every self-play policy of this process, set by ``load_weights``.
_model = None
_actor = None


def own_observation(
    fighters: Mapping[str, FighterState], fighter_id: str, config: Config
) -> list[float]:
    """Return the vector a network sees when it plays ``fighter_id``: itself, then its enemies."""
    structured = get_observation(fighters[fighter_id], enemies_of(fighter_id, fighters), config)
    return encode_observation(structured, config)


def weights_of(model):
    """Return all of a model's parameters as one flat float32 numpy array.

    One array is one device transfer, and it crosses a process pipe as plain bytes. Torch
    tensors sent through a pipe would be moved into shared memory instead.
    """
    from torch.nn.utils import parameters_to_vector

    return parameters_to_vector(model.parameters()).detach().cpu().numpy()


def load_weights(vector, config: Config) -> None:
    """Load a flat weight vector (see ``weights_of``) into this process's self-play network."""
    global _model, _actor
    import torch
    from torch.nn.utils import vector_to_parameters

    from arena_ai.agents.policy import ActorCritic
    from arena_ai.agents.stacked_actor import StackedActor

    if _model is None:
        torch.set_num_threads(1)  # A tiny network per step: threads only add overhead.
        _model = ActorCritic(config)
        _actor = StackedActor(_model, torch.device("cpu"))
    vector_to_parameters(torch.tensor(vector, dtype=torch.float32), _model.parameters())
    _actor.load(_model)


class SelfPlayPolicy:
    """An opponent that plays with the learner's current weights."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._generator = None

    def resample(self, rng) -> None:
        """Seed the move sampling from the environment's generator, at each episode reset.

        Without it the moves would come from the process's global generator, so a seeded
        run would not repeat.
        """
        import torch

        self._generator = torch.Generator().manual_seed(rng.getrandbits(63))

    def __call__(self, observation: list[float]) -> int:
        if _actor is None:
            raise RuntimeError("self-play opponent has no weights: call set_opponent_weights")
        import torch

        from arena_ai.agents.distribution import sample_with_log_prob

        with torch.no_grad():
            logits, _ = _actor(torch.tensor([observation], dtype=torch.float32))
            action, _ = sample_with_log_prob(logits, self._generator)
        return int(action.item())


def make_opponent_policy(name: str, config: Config) -> EpisodePolicy | SelfPlayPolicy:
    """Return the policy for a training opponent name: a scripted one, or ``agent``."""
    if name == SELF_PLAY:
        return SelfPlayPolicy(config)
    return EpisodePolicy(name, config)
