# Runtime configuration and deployment defaults

The Docker image copies `backend/` and `example.env`; it does not copy the
repository's `config.env`. Settings searches for that file relative to the
installed module, so an absent file leaves class defaults in effect. Process
environment variables take precedence. A checkout can consequently behave
differently from the deployed image.

During the September 2026 incident review, both Render services were checked:
neither defined model, Vertex, deep-research, or Agents SDK overrides. The
following differences therefore matter for this deployment. Recheck service
configuration before treating this observation as current in a later incident.

| Setting | Image default after this fix | Repository config.env | Consequence of copying the whole file |
| --- | --- | --- | --- |
| OPENAI_PM_MODEL | gpt-5.5 | gpt-5.5 | No further change; the former class default was gpt-5.5-pro. |
| DEEP_RESEARCH_MAX_ROUNDS | 1 | 3 | More possible critique/research rounds. |
| DEEP_RESEARCH_MAX_QUESTIONS_PER_ROUND | 2 | 3 | Ordinary follow-up question ceiling rises from 2 to 9 across rounds. |
| USE_AGENTS_SDK | false | true | Changes execution routing where the SDK flag is used. |
| VERTEX_PROJECT_ID | empty | project configured | Enables Vertex routing and requires its credentials. |
| VERTEX_MODEL | empty | gemini-2.5-pro | Changes the fallback Gemini model when no explicit caller model is supplied. |
| ENABLE_LIVE_DATA | false | true | No effective change here: Render explicitly sets true on both services. |

The question ceiling is not a multiplier for total memo cost: early stopping
can reduce work, and seeded review questions have separate behavior. Explicit
caller models take precedence over `VERTEX_MODEL`; news and social currently
pass their Flash model explicitly. Vertex still changes their backend and
authentication. With neither Gemini API credentials nor Vertex configured,
Gemini specialists remain unavailable; an OpenAI model fix does not enable them.

This repair changes only the PM class default to the model already selected
by the repository for the existing Chat Completions path. It does not copy
`config.env`, activate extra research, or change prompts or rating semantics.
Enabling those other differences requires a separate owner decision about
execution routing, credentials, and spend.

`test_config_image_defaults.py` imports the real configuration module from a
temporary image-shaped directory in a new process with no inherited environment
or env files. It checks the model, bounded research defaults, disabled SDK and
Vertex, and explicit process-environment override precedence without provider calls.
