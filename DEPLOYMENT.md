# CI/CD deployment to the existing Azure VM

The application runs as a Docker Compose stack on the Linux VM `AgenticAI`.
nginx and Let's Encrypt remain on the VM and proxy public HTTPS traffic to
`127.0.0.1:8000`. No Azure Container Registry or additional Azure service is
required.

## Delivery flow

1. Every pull request and push runs dependency checks, compilation, linting,
   unit/integration tests, PostgreSQL migrations, Redis connectivity, and a
   complete Docker build.
2. After a successful push to `main`, GitHub Actions creates a release archive
   directly from the tested Git commit.
3. The archive is copied to the existing VM through SSH.
4. The VM builds `agentic-logistics:<git-sha>` locally and updates Docker
   Compose.
5. The deployment polls `/health` for up to 90 seconds.
6. If the new release fails, Docker Compose returns to the previously recorded
   image and release definition.

Manual execution through `workflow_dispatch` is also allowed for `main`.

## Repository files

- `.github/workflows/ci.yml` — CI and production deployment
- `deploy/docker-compose.production.yml` — production stack
- `deploy/deploy_azure_vm.sh` — release, health-check, and rollback logic

## Required GitHub configuration

Create the GitHub environment `production`. A required reviewer is optional but
recommended for a portfolio production deployment.

Add these repository or `production` environment secrets:

- `AZURE_VM_HOST` — public IP or DNS name of the VM;
- `AZURE_VM_USER` — Linux user used by the existing SSH connection;
- `AZURE_VM_SSH_KEY` — private key dedicated to deployment;
- `AZURE_VM_KNOWN_HOSTS` — verified SSH host-key entry for the VM.

Optional GitHub Actions variables:

- `AZURE_VM_SSH_PORT` — defaults to `22`;
- `AZURE_DEPLOYMENT_PATH` — for the existing VM use
  `/home/azureuser/Agentic-Logistics-Optimizer`; otherwise it defaults to
  `$HOME/agentic-logistics`.

The Azure Resource Group name is not needed by this simpler workflow.

### Obtain the known-hosts entry

Run this from a trusted computer and compare the fingerprint with the VM before
adding the complete output as `AZURE_VM_KNOWN_HOSTS`:

```bash
ssh-keyscan -p 22 <vm-ip-or-dns>
```

Do not use `StrictHostKeyChecking=no`. Keeping a verified host key in GitHub
prevents the workflow from silently connecting to an impersonated server.

## VM prerequisites

The existing VM user must be able to connect through SSH and run Docker without
an interactive password. The VM also needs:

- Docker Engine;
- Docker Compose v2 (`docker compose`);
- curl;
- tar;
- nginx forwarding to `http://127.0.0.1:8000`.

The workflow does not modify nginx, firewall rules, or Azure resources.

## Production environment file

The existing VM already uses
`/home/azureuser/Agentic-Logistics-Optimizer/.env`; the deployment reuses it and
never sends its contents to GitHub Actions. For a new VM, create the file once:

```bash
mkdir -p "$HOME/agentic-logistics"
nano "$HOME/agentic-logistics/.env"
chmod 600 "$HOME/agentic-logistics/.env"
```

Minimum configuration:

```dotenv
ENVIRONMENT=production
POSTGRES_USER=agentic_user
POSTGRES_PASSWORD=<strong-database-password>
POSTGRES_DB=logistics_app

LLM_API_KEY=<gemini-api-key>
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.6-flash
LLM_MAX_TOKENS=2048
LLM_REASONING_EFFORT=low

API_KEY=<random-api-key>
PORTFOLIO_PASSWORD=<portfolio-password>
CORS_ORIGINS=https://gwprojects.switzerlandnorth.cloudapp.azure.com

LLM_TEMPERATURE=0.3
LLM_TOP_P=0.9
LLM_MAX_TOKENS=1024
LLM_REQUEST_TIMEOUT_SECONDS=20
ANALYSIS_TIMEOUT_SECONDS=75
HEALTH_CHECK_TIMEOUT_SECONDS=2
```

The deployment never uploads, replaces, or prints this file. Production Compose
constructs `DATABASE_URL` from the three PostgreSQL variables, matching the
existing deployment.

## Release layout on the VM

```text
/home/azureuser/Agentic-Logistics-Optimizer/
├── .env                  # production secrets
├── .deployed-image       # last healthy Docker image
├── .deployed-release     # last healthy release directory
├── current -> releases/… # symlink to the active source revision
├── chroma_db/            # persistent vector data
└── releases/
    └── <git-sha>/        # immutable source used to build the image
```

PostgreSQL and Redis use persistent named Docker volumes. Releases never delete
those volumes or the ChromaDB directory.

## Existing deployment

Before the first automated deployment, inspect the currently running Compose
project and its volumes. This VM uses the fixed project name
`agentic-logistics-optimizer`. The new pipeline keeps that project name, the
existing container names, PostgreSQL/Redis volumes, `.env`, and ChromaDB
directory. On its first run it also records the current image and Compose file,
so a failed release can return to the manually deployed version. A database
backup before the first cutover is still prudent.

## Operations

Show the active release:

```bash
cat "$HOME/Agentic-Logistics-Optimizer/.deployed-image"
readlink -f "$HOME/Agentic-Logistics-Optimizer/current"
```

Show containers:

```bash
cd "$HOME/Agentic-Logistics-Optimizer/current/deploy"
IMAGE_REF="$(cat ../../.deployed-image)" \
CHROMA_DATA_PATH="$HOME/Agentic-Logistics-Optimizer/chroma_db" \
docker compose --project-name agentic-logistics-optimizer \
  --env-file ../../.env \
  --file docker-compose.production.yml \
  ps
```

Application logs use the same command prefix followed by:

```bash
logs --tail 200 app
```
