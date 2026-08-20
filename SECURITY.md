# Security Policy

PhysioRAG is a research prototype and **not a certified medical device**. It must
not be used for clinical decision-making or patient diagnosis.

## Reporting a vulnerability

If you discover a security issue, please report it privately rather than opening
a public issue:

- Use GitHub's **"Report a vulnerability"** (Security Advisories) on this
  repository, or
- Contact the maintainer via [LinkedIn](https://www.linkedin.com/in/joergbahlmann).

Please include a description, reproduction steps, and the potential impact.
We aim to acknowledge reports within a reasonable time and will coordinate a fix
and disclosure timeline with you.

## Scope and expectations

- **No secrets in the repo.** PhysioNet credentials belong in a local, untracked
  `.env` (see `.env.example`); never commit them.
- **Offline by design.** PhysioRAG is meant to run air-gapped. An air gap is a
  strong security control but not, by itself, a GDPR/HIPAA compliance
  certification — production deployments still require organizational measures,
  documented risk assessments, access controls, and audits.
- **Third-party components** (model weights, datasets, Weaviate, Ollama) have
  their own security posture and licenses; review them for your environment.
