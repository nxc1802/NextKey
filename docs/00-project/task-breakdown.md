# Task Breakdown

This file is the execution map for the full project. Each task is split into
phases with concrete exit criteria.

## Task 0: Project Foundation

Goal: create a reproducible engineering workspace.

Phases:

- 0.1 initialize git and base repository layout
- 0.2 create Python environment and dependency management
- 0.3 define data, model, report, and experiment artifact folders
- 0.4 add command entry points for data generation, training, evaluation, and demo
- 0.5 add formatting and lightweight checks

Done when:

- a new developer can run setup instructions
- empty-but-valid commands exist for `data`, `train`, `eval`, and `demo`
- generated artifacts are excluded from git by default

## Task 1: Scope And Research Contract

Goal: lock the problem definition without prematurely locking the model.

Phases:

- 1.1 define final problem statement and use cases
- 1.2 finalize research questions
- 1.3 classify must-have, should-have, and extension features
- 1.4 define acceptance criteria for MVP and final defense
- 1.5 write a short proposal document

Done when:

- the project can be explained as compact Vietnamese input restoration, not only
  diacritic restoration
- the model-selection task remains open and benchmark-driven

## Task 2: Data System

Goal: build the dataset pipeline before serious model training.

Phases:

- 2.1 define canonical JSONL schema
- 2.2 collect and normalize clean Vietnamese source sentences
- 2.3 implement synthetic noise generator
- 2.4 create dataset QA filters
- 2.5 create train/dev/test splits
- 2.6 run human-refined annotation pilot
- 2.7 freeze `dev_human` and `test_human` sets for evaluation

Done when:

- at least 100k synthetic pairs exist
- at least 1k human-refined samples exist for evaluation
- split leakage checks pass

## Task 3: Evaluation Harness

Goal: make all model comparisons fair and repeatable.

Phases:

- 3.1 implement NLP metrics
- 3.2 implement typo and overcorrection metrics
- 3.3 implement top-k candidate metrics
- 3.4 implement edge latency and memory benchmark
- 3.5 generate experiment reports from structured result files

Done when:

- every candidate model can be evaluated through the same command
- metric output is saved as machine-readable files and human-readable tables

## Task 4: Model Selection Open Gate

Goal: choose model roles using evidence from this project dataset.

Phases:

- 4.1 define candidate model pool
- 4.2 run small smoke fine-tunes
- 4.3 compare candidates on the same synthetic and human dev splits
- 4.4 choose teacher candidate
- 4.5 choose student/edge candidate
- 4.6 choose whether a reranker is worth keeping

Done when:

- model choice is justified by metrics, latency, memory, and implementation cost
- unresolved candidates are explicitly marked as dropped or deferred

## Task 5: Core Restoration Pipeline

Goal: produce a usable restoration system.

Phases:

- 5.1 implement rule-based baseline
- 5.2 implement char-level seq2seq baseline
- 5.3 fine-tune selected pretrained candidates
- 5.4 add top-k beam candidate generation
- 5.5 add post-processing and keep-list policy
- 5.6 run error analysis by domain and noise type

Done when:

- the selected model clearly beats rule and small char baselines on human dev
- top-k candidates are stable enough for demo use

## Task 6: API And Demo

Goal: expose the model through a demonstrable product surface.

Phases:

- 6.1 build `/restore`
- 6.2 build candidate panel
- 6.3 build diff view
- 6.4 build feedback logging endpoint
- 6.5 build benchmark/status view
- 6.6 write a repeatable demo script

Done when:

- a user can enter compact text and see top-k restored candidates
- accepted, rejected, and edited outputs can be logged in a controlled format

## Task 7: Edge Optimization

Goal: create a small deployable model or a credible edge prototype.

Phases:

- 7.1 create teacher output dataset for distillation
- 7.2 train student model
- 7.3 export to ONNX first
- 7.4 quantize to INT8 where feasible
- 7.5 benchmark CPU and target device
- 7.6 document quality and latency trade-off

Done when:

- a student model has measured size, RAM, p50 latency, and p95 latency
- the quality loss versus teacher is quantified

## Task 8: Personalization And Privacy

Goal: add personalization without creating unsafe data handling.

Phases:

- 8.1 implement personal dictionary
- 8.2 implement phrase/domain priors
- 8.3 add personalized reranking features
- 8.4 pilot LoRA per-user if enough data exists
- 8.5 pilot DPO-lite only after chosen/rejected pairs exist
- 8.6 add consent, deletion, and retention rules

Done when:

- base versus personalized behavior is measurable
- privacy controls are visible in the demo and documented

## Task 9: Final Evaluation And Defense

Goal: package the project into a defensible graduation deliverable.

Phases:

- 9.1 run final baselines
- 9.2 run ablation studies
- 9.3 run edge benchmark
- 9.4 run user study pilot
- 9.5 write thesis chapters
- 9.6 prepare slides, video, and demo script

Done when:

- final claims are backed by tables, logs, screenshots, and repeatable commands
