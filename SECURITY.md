# Security model

## Secrets

Application secrets are never copied into the Docker image or committed to the
repository. Production values live only in `$HOME/agentic-logistics/.env` on
the Azure VM with mode `600`.

Sensitive application values include:

- `GITHUB_TOKEN`
- `API_KEY`
- `PORTFOLIO_PASSWORD`
- `POSTGRES_PASSWORD`
- `DATABASE_URL`
- optional `LANGSMITH_API_KEY`

`.env`, local virtual environments, ChromaDB data, and test artifacts are
excluded by `.gitignore` and `.dockerignore`.

## Deployment access

GitHub Actions uses a dedicated SSH private key stored as the protected secret
`AZURE_VM_SSH_KEY`. The corresponding public key should belong to a dedicated
deployment account or a restricted key entry on the existing Azure VM.

The workflow also requires a verified `known_hosts` entry. It does not disable
SSH host verification and therefore does not silently trust a changed server
identity. The deployment account needs Docker access but should not have a
general-purpose Azure account or Azure subscription credentials.

The `production` GitHub environment should require a reviewer. Rotate the
deployment key if a maintainer or repository integration loses authorization.

## Network boundaries

- Only nginx exposes ports 80 and 443 publicly.
- FastAPI binds to `127.0.0.1:8000` on the VM.
- PostgreSQL and Redis exist only on the private Docker network.
- Their ports are not published by the production Compose file.
- Azure NSG rules should restrict SSH to the addresses that genuinely need it.

## API protections

- `/analyze`, `/batch-analyze`, and `/debug/llm-test` require `x-api-key`.
- The portfolio page uses Basic Auth and must be accessed only through HTTPS.
- `/demo/analyze` accepts predefined scenarios rather than an unrestricted
  public prompt.
- Pydantic validates request and agent response structures.
- LLM calls and the complete workflow have bounded timeouts.
- Carrier pricing comes from a deterministic typed tool instead of the LLM.

The current application rate-limit dependency remains permissive. A public
deployment should additionally enforce request limits in nginx or an API
gateway.

## CI/CD controls

- Pull requests cannot deploy production.
- CI validates dependencies, PostgreSQL migrations, Redis connectivity, Python
  code, tests, and the Docker image.
- `git archive` packages only files committed in the revision that passed CI.
- Production images are tagged with the immutable Git commit SHA.
- Failed health checks restore the previous image and Compose definition.
- The pipeline never overwrites `.env` or deletes persistent data volumes.

## Incident response

If a secret is exposed:

1. revoke or rotate it immediately;
2. update the protected VM `.env` or GitHub secret;
3. restart or redeploy the affected service;
4. inspect GitHub Actions, SSH, nginx, and application logs;
5. remove the value from Git history while treating it as permanently
   compromised.

Before pushing, inspect staged changes and run secret scanning:

```bash
git diff --cached
git grep -n -E "(github_pat_|ghp_|BEGIN .*PRIVATE KEY|POSTGRES_PASSWORD=.+)"
```

These patterns are an additional guard, not a replacement for GitHub secret
scanning or a dedicated security scanner.
