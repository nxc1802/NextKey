# Roadmap

This roadmap assumes a 16-20 week graduation-project window. It is milestone
based, not date based, so it can be shifted when the academic schedule changes.

## Milestone 1: Project Contract

Target: weeks 1-2

Deliverables:

- problem statement
- research questions
- MVP scope
- documentation structure
- initial repository structure

Exit gate:

- project is positioned as compact Vietnamese input restoration
- model choice remains explicitly open

## Milestone 2: Data Foundation

Target: weeks 3-5

Deliverables:

- dataset schema
- noise taxonomy
- synthetic generator v1
- clean source corpus v1
- dataset QA report

Exit gate:

- at least 100k synthetic pairs
- representative examples for each core noise type

## Milestone 3: Baselines And Evaluation

Target: weeks 6-8

Deliverables:

- rule baseline
- char-level baseline
- first pretrained candidate smoke run
- metric scripts
- first benchmark table

Exit gate:

- all candidates can be evaluated through the same harness
- baseline results are reproducible

## Milestone 4: Model Selection

Target: weeks 9-10

Deliverables:

- model-selection report
- teacher candidate decision
- student candidate decision
- reranker decision

Exit gate:

- chosen model roles are backed by human-dev results and resource estimates

## Milestone 5: Demo MVP

Target: weeks 11-12

Deliverables:

- restoration API
- top-k candidate response
- demo UI
- feedback event schema

Exit gate:

- compact input can be restored through an end-to-end demo path

## Milestone 6: Edge Or Personalization Push

Target: weeks 13-15

Preferred path:

- distillation
- ONNX export
- quantization
- edge benchmark

Fallback path:

- personal dictionary
- feedback logging
- base versus personalized reranking comparison

Exit gate:

- at least one advanced contribution is measured and demo-visible

## Milestone 7: Final Evaluation

Target: weeks 16-18

Deliverables:

- final baseline table
- ablation table
- error analysis
- edge or personalization result
- user study pilot if possible

Exit gate:

- final claims are supported by logs and frozen splits

## Milestone 8: Defense Package

Target: weeks 19-20

Deliverables:

- thesis
- slide deck
- demo video
- live demo checklist
- final risk and limitation section

Exit gate:

- the project can be presented even if network, GPU, or one optional extension
  fails during defense
