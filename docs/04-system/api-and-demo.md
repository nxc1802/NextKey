# API And Demo Specification

The demo should prove the full loop: compact input, model candidates, user
choice, feedback, and measurable latency.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/restore` | POST | Restore compact input and return top-k candidates |
| `/feedback` | POST | Save accepted, rejected, or edited output |
| `/personalize/train` | POST | Trigger optional personalization update |
| `/metrics/edge` | GET | Return model size, latency, RAM, and runtime info |
| `/privacy/delete` | POST | Delete user personalization data |

## `/restore` Request

```json
{
  "input": "t muonviet 1 bai ve ai cho svnam4",
  "top_k": 5,
  "user_id_hash": "optional",
  "mode": "base"
}
```

## `/restore` Response

```json
{
  "request_id": "req_001",
  "best": "Tôi muốn viết một bài về AI cho sinh viên năm 4.",
  "candidates": [
    {
      "text": "Tôi muốn viết một bài về AI cho sinh viên năm 4.",
      "rank": 1,
      "score": 0.93
    }
  ],
  "model": {
    "name": "open-selection",
    "version": "unset"
  },
  "latency_ms": 142
}
```

## Demo UI Areas

| Area | Purpose |
|---|---|
| Input box | Enter or paste compact text |
| Candidate panel | Show top-3 or top-5 candidates |
| Diff view | Highlight model edits |
| Feedback controls | Accept, reject, or edit |
| Personalization toggle | Enable/disable user adaptation |
| Privacy controls | View/delete/export logs |
| Benchmark panel | Show latency, model size, runtime mode |

## Feedback Event

```json
{
  "request_id": "req_001",
  "compact_input": "t muonviet 1 bai ve ai cho svnam4",
  "candidates": [
    "Tôi muốn viết một bài về AI cho sinh viên năm 4.",
    "Tôi muốn viết một bài về ai cho sinh viên năm 4."
  ],
  "chosen_candidate": "Tôi muốn viết một bài về AI cho sinh viên năm 4.",
  "edited_output": null,
  "event_type": "accept",
  "consent_scope": "personalization_only"
}
```

## Demo Acceptance Criteria

- restore request completes reliably for short and medium sentences
- top-k candidates are visible
- user can accept, reject, or edit
- feedback writes to structured logs
- benchmark panel reports latency
- demo can run with a fallback baseline if the main model is unavailable
