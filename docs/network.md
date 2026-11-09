# The neural network

This doc shows how the agent decides, and how that decision is trained. Read the diagrams
top to bottom to follow a decision, then bottom to top to follow the gradients. For the
training loop around it see [training-process.md](training-process.md). For the layout of
the input see [simulation-spec.md](simulation-spec.md#observation).

Code: [`agents/policy.py`](../src/arena_ai/agents/policy.py) (`ActorCritic`,
`AttentionNetwork`), [`agents/attention.py`](../src/arena_ai/agents/attention.py)
(`EntityAttention`), [`agents/distribution.py`](../src/arena_ai/agents/distribution.py)
(the action heads) and [`training/ppo.py`](../src/arena_ai/training/ppo.py) (the loss).
Sizes are the defaults in [`configs/agent.yaml`](../configs/agent.yaml): `embed_dim` 64,
`num_heads` 4, `hidden_layers` [128, 128], and `observation.max_opponents` 3.

## Two networks, one input

`ActorCritic` holds two separate networks with **no shared parameters**. Both read the same
observation. The **policy network** (the actor) outputs 9 logits that become the action.
The **value network** (the critic) outputs one number, the expected return, which PPO uses
to judge how good an action was. Each has its own attention block.

```mermaid
flowchart LR
    OBS["Observation vector<br/>59 values"] --> PN["Policy network<br/>attention block, hidden layers, output"]
    OBS --> VN["Value network<br/>attention block, hidden layers, output"]
    PN --> LOGITS["9 logits"]
    LOGITS --> DIST["4 independent action heads<br/>move, jump, attack, parry"]
    DIST --> ACT["One of 24 actions"]
    VN --> VAL["V(s): one number"]
    VAL -. "critic target" .-> LOSS["PPO loss<br/>see Training"]
    DIST -. "log probability" .-> LOSS
```

## What the agent sees

The input is one flat vector from the acting fighter's point of view (details and scaling in
the [simulation spec](simulation-spec.md#observation)). `F` is 22 features per fighter and
`K` is the number of opponent slots (3).

```mermaid
flowchart TB
    subgraph IN["Observation: 14 x (1 + 3) + 3 = 59 values"]
        direction LR
        S["Own row (F = 22)<br/>x, y, vx, vy, on_ground, facing,<br/>health, attack_*_left (3), attack_counter,<br/>parry_*_left (2), parry_window, counter_frames,<br/>stagger_frames, parry_held, busy_frames_left,<br/>attack_type (one-hot),<br/>rel_x_ahead, rel_y, rel_x_aimed"]
        O1["Opponent slot 0 (F = 22)"]
        O2["Opponent slot 1 (F = 22)"]
        O3["Opponent slot 2 (F = 22)"]
        M["Mask (K = 3)<br/>1 = living opponent<br/>0 = absent or dead"]
    end
    NOTE["A missing or dead opponent is all zeros with mask 0.<br/>Opponents are sorted by id, so slot order carries no meaning."]
    IN -.-> NOTE
```

## Inside one network

The same structure is used twice, once for the policy and once for the value. Only the
output layer differs (9 logits, or 1 value). Every box labelled "Linear" is a weight matrix
plus a bias; every parameter in the diagram is trained.

```mermaid
flowchart TB
    OBS["Observation, 59 values"] --> SPLIT["Split"]
    SPLIT -->|"own row [B, 14]"| PHIS
    SPLIT -->|"opponent rows [B, 3, 14]"| PHIO
    SPLIT -->|"mask [B, 3]"| MASK

    subgraph ATT["Attention block (EntityAttention), embed_dim d = 64"]
        direction TB
        PHIS["phi_self<br/>Linear 14 to 64, ReLU, Linear 64 to 64"]
        PHIO["phi_opp, shared by every opponent<br/>Linear 14 to 64, ReLU, Linear 64 to 64"]
        PHIS --> ES["e_self [B, 64]"]
        PHIO --> EO["e_1 ... e_3 [B, 3, 64]"]
        ES --> ENT["Entities E = e_self, e_1, e_2, e_3<br/>[B, 4, 64]"]
        EO --> ENT
        ES --> WQ["W_Q: Linear 64 to 64"]
        WQ --> Q["Query q<br/>only the acting fighter asks"]
        ENT --> WK["W_K: Linear 64 to 64"]
        ENT --> WV["W_V: Linear 64 to 64"]
        WK --> K["Keys, one per entity"]
        WV --> V["Values, one per entity"]
        Q --> SC["Scores = q . K / sqrt(d_k)<br/>one per entity [B, 1, 4]"]
        K --> SC
        MASK["Mask"] --> SC
        SC -->|"absent or dead opponents set to -inf<br/>own slot is never masked"| SM["Softmax: attention weights"]
        SM --> CTX["Context = weights x V [B, 64]<br/>(with several heads: join the heads,<br/>then an output projection Linear 64 to 64)"]
        V --> CTX
    end

    ES --> CAT["Concatenate: e_self then context<br/>[B, 128]"]
    CTX --> CAT
    CAT --> H1["Hidden 1: Linear 128 to 128, ReLU"]
    H1 --> H2["Hidden 2: Linear 128 to 128, ReLU"]
    H2 --> OUT["Output layer: Linear 128 to 9 (policy) or 1 (value)"]
    OUT --> RES["Logits (policy) or V(s) (value)"]
```

How to read it:

- **Embeddings.** `phi_self` turns the acting fighter's 22 numbers into a 64-number summary,
  `e_self`. `phi_opp` does the same for each opponent with the **same weights**, so what is
  learned about one opponent applies to all of them.
- **Query, keys, values.** Only the acting fighter has a query. Every entity (itself and each
  opponent) has a key and a value. A score is how well the fighter's query matches an
  entity's key. After the softmax the scores become weights that sum to 1, and the
  **context** is the weighted sum of the values: a blend that leans toward the entities the
  fighter "attends to".
- **Heads.** With `num_heads` above 1 the 64 numbers of the query, keys and values are
  split into slices (4 heads: 16 numbers each). Every head runs its own softmax over all the
  entities, so one head can lean toward the nearest opponent and another toward the one
  whose swing is about to land. The heads' outputs are joined and mixed by an output
  projection. The split does not change the parameter count.
- **Mask.** An absent or dead opponent gets a score of minus infinity, so its weight is
  exactly 0 and it has no influence, whatever its (zeroed) row holds.
- **Concatenate.** The hidden layers receive `[e_self ; context]` side by side, so the
  fighter's own state reaches them directly and is not only one entry in an average.
- **Limits.** The context is a blend, not each opponent separately: two opponents that
  swing at once can have their frame countdowns averaged within a head, which is why there
  are several heads. Each row carries `rel_x_ahead`, `rel_y` and `rel_x_aimed` (position
  relative to the acting fighter), so the network does not have to subtract two absolute
  positions to tell a near threat from a far one.

## The decision

The 9 policy logits are split into four independent categorical distributions
([`FactoredCategorical`](../src/arena_ai/agents/distribution.py)). The action index is built
from one choice per head, and its log probability is the sum of the four heads' log
probabilities.

```mermaid
flowchart LR
    L["9 logits"] --> HM["move head<br/>3 logits: none, left, right"]
    L --> HJ["jump head<br/>2 logits"]
    L --> HA["attack head<br/>2 logits"]
    L --> HP["parry head<br/>2 logits"]
    HM --> SM1["softmax, sample"]
    HJ --> SM2["softmax, sample"]
    HA --> SM3["softmax, sample"]
    HP --> SM4["softmax, sample"]
    SM1 --> IDX["action index =<br/>move + 3 jump + 6 attack + 12 parry<br/>(24 actions)"]
    SM2 --> IDX
    SM3 --> IDX
    SM4 --> IDX
```

During training the action is sampled. In the viewer the trained agent plays the most likely
choice of each head ("Best move") or samples ("Varied").

## Training

[`train_step`](../src/arena_ai/training/ppo.py) runs once per minibatch of the rollout. The
rollout supplies, for each step, the observation, the action taken, the log probability it
had when taken (`old log prob`), the **advantage** (how much better than expected, from GAE
and normalized) and the **return** (the discounted reward to go).

```mermaid
flowchart TB
    B["Minibatch: observations, actions, old log probs, advantages, returns"] --> FWD["Forward pass through both networks"]
    FWD --> LG["Policy logits"]
    FWD --> VP["Value predictions V(s)"]
    LG --> NLP["New log prob of the stored action<br/>sum over the 4 heads"]
    LG --> ENT["Entropy of the 4 heads"]
    NLP --> RATIO["Ratio = exp(new log prob - old log prob)"]
    RATIO --> CLIP["Clipped surrogate<br/>min(ratio x A, clip(ratio, 0.8, 1.2) x A)"]
    CLIP --> PL["Policy loss = - mean of the surrogate"]
    VP --> VL["Value loss = mean squared error<br/>(V(s) - return)"]
    ENT --> EB["Entropy bonus"]
    PL --> TOT["Total loss =<br/>policy loss + 0.5 x value loss - 0.011 x entropy"]
    VL --> TOT
    EB --> TOT
    TOT --> BP["loss.backward()"]
    BP --> GC["Clip the gradient norm to 0.5"]
    GC --> OPT["Adam step"]
    OPT --> W["Updated weights"]
```

### Where the gradients go

`loss.backward()` sends gradients back along every arrow above, in reverse:

| Loss term | Reaches | Through |
| --------- | ------- | ------- |
| Policy loss (via the ratio) and entropy bonus | the **policy network** only | the output layer, the hidden layers, then the attention block: `W_V` and `W_K` (through the context and the scores), `W_Q`, `phi_opp` (through the keys and values of the opponents), and `phi_self` (both directly through `e_self` and through the query, key and value of the acting fighter) |
| Value loss | the **value network** only | the same path in its own copy of the layers |

The two networks share no parameters, so the value loss never changes the policy and the
other way round. The advantage and return are fixed numbers in this step: they come from the
rollout and carry no gradient. A masked opponent has weight 0, so it receives no gradient
either.

### Why each piece is there

- **Clipped ratio.** It stops one update moving the policy far from the one that collected
  the data. `clip_range` is 0.2.
- **Value loss.** It trains the critic to predict the return, so the advantage (return minus
  prediction, smoothed by GAE with `discount_factor` 0.99 and `gae_lambda` 0.95) is a fair
  signal. `value_loss_coef` is 0.5.
- **Entropy bonus.** It keeps the action distributions from collapsing too early, so the
  agent keeps exploring. `entropy_coef` is 0.011. It is subtracted because the optimizer
  minimizes.
- **Gradient clipping.** It limits how large one update can be (norm 0.5).

## Checkpoints

A checkpoint stores the weights, the optimizer state and a `signature` (the observation
`spec_version`, the features per fighter, `hidden_layers`, `embed_dim`, `num_heads` and the
activations). Loading checks it and names each difference. `max_opponents` is not in the
signature, because no weight depends on it: a network trained with 3 opponent slots loads
with 5.

## Size

With the defaults the network has 112,778 parameters: 56,905 in the policy network and 55,873
in the value network.
