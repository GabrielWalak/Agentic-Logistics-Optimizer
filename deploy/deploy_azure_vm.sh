#!/usr/bin/env bash
set -Eeuo pipefail

RELEASE_ID="${1:?Git commit SHA is required}"
REQUESTED_DEPLOYMENT_PATH="${2:-}"
ARCHIVE_PATH="${3:?Release archive path is required}"

if [[ ! "$RELEASE_ID" =~ ^[0-9a-f]{40}$ ]]; then
  printf '[deploy] Invalid Git commit SHA\n' >&2
  exit 1
fi

DEPLOYMENT_PATH="${REQUESTED_DEPLOYMENT_PATH:-$HOME/agentic-logistics}"
RELEASES_PATH="$DEPLOYMENT_PATH/releases"
RELEASE_PATH="$RELEASES_PATH/$RELEASE_ID"
ENV_FILE="$DEPLOYMENT_PATH/.env"
CHROMA_DATA_PATH="$DEPLOYMENT_PATH/chroma_db"
IMAGE_REF="agentic-logistics:$RELEASE_ID"
IMAGE_MARKER="$DEPLOYMENT_PATH/.deployed-image"
RELEASE_MARKER="$DEPLOYMENT_PATH/.deployed-release"
COMPOSE_MARKER="$DEPLOYMENT_PATH/.deployed-compose"
CURRENT_LINK="$DEPLOYMENT_PATH/current"

log() {
  printf '[deploy] %s\n' "$1"
}

compose() {
  local image_ref="$1"
  local compose_file="$2"
  shift 2
  IMAGE_REF="$image_ref" CHROMA_DATA_PATH="$CHROMA_DATA_PATH" docker compose \
    --project-name agentic-logistics-optimizer \
    --project-directory "$(dirname "$compose_file")" \
    --env-file "$ENV_FILE" \
    --file "$compose_file" \
    "$@"
}

wait_for_application() {
  local attempts=18
  local health_payload
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    health_payload="$(
      curl --fail --silent --show-error http://127.0.0.1:8000/health || true
    )"
    if [[ "$health_payload" == *'"status":"healthy"'* ]]; then
      return 0
    fi
    sleep 5
  done
  return 1
}

log "Validating VM prerequisites"
command -v docker >/dev/null
docker compose version >/dev/null
command -v curl >/dev/null
command -v tar >/dev/null

install -d -m 0755 "$DEPLOYMENT_PATH" "$RELEASES_PATH" "$CHROMA_DATA_PATH"
if [[ ! -f "$ENV_FILE" ]]; then
  log "Missing $ENV_FILE. Create it before the first deployment."
  exit 1
fi
chmod 600 "$ENV_FILE"

if [[ ! -f "$ARCHIVE_PATH" ]]; then
  log "Release archive does not exist: $ARCHIVE_PATH"
  exit 1
fi
trap 'rm -f "$ARCHIVE_PATH"' EXIT

if [[ ! -d "$RELEASE_PATH" ]]; then
  staging_path="$(mktemp -d "$RELEASES_PATH/.incoming-$RELEASE_ID-XXXXXX")"
  tar --extract --gzip --file "$ARCHIVE_PATH" --directory "$staging_path"
  mv "$staging_path" "$RELEASE_PATH"
fi

COMPOSE_FILE="$RELEASE_PATH/deploy/docker-compose.production.yml"
if [[ ! -f "$COMPOSE_FILE" || ! -f "$RELEASE_PATH/Dockerfile" ]]; then
  log "Release is missing Dockerfile or production Compose definition"
  exit 1
fi

# Compose resolves env_file relative to the release. The symlink keeps secrets
# outside immutable release directories and outside the Git repository.
ln -sfn "$ENV_FILE" "$RELEASE_PATH/deploy/.env"

log "Validating the production Compose definition"
compose "$IMAGE_REF" "$COMPOSE_FILE" config --quiet

log "Building immutable image $IMAGE_REF"
docker build --tag "$IMAGE_REF" "$RELEASE_PATH"

previous_image=""
previous_compose=""
if [[ -f "$IMAGE_MARKER" ]]; then
  previous_image="$(<"$IMAGE_MARKER")"
fi
if [[ -f "$COMPOSE_MARKER" ]]; then
  previous_compose="$(<"$COMPOSE_MARKER")"
fi

# The first automated release can still roll back to the manually deployed
# Compose stack that existed before CI/CD markers were introduced.
if [[ -z "$previous_image" ]] && docker inspect agentic-logistics-api >/dev/null 2>&1; then
  previous_image="$(
    docker inspect agentic-logistics-api --format '{{.Config.Image}}'
  )"
fi
if [[ -z "$previous_compose" ]] && docker inspect agentic-logistics-api >/dev/null 2>&1; then
  previous_compose="$(
    docker inspect agentic-logistics-api \
      --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}'
  )"
fi

rollback() {
  if [[ -z "$previous_image" || ! -f "$previous_compose" ]]; then
    log "No previous release is available for rollback"
    return 1
  fi

  log "Rolling back to $previous_image"
  compose "$previous_image" "$previous_compose" up --detach --remove-orphans
  if wait_for_application; then
    log "Rollback completed; the deployment remains failed"
    return 0
  fi

  log "Rollback health check also failed"
  return 1
}

log "Starting the production stack"
if ! compose "$IMAGE_REF" "$COMPOSE_FILE" up --detach --remove-orphans; then
  log "Compose failed before the application health check"
  rollback || true
  exit 1
fi

if wait_for_application; then
  printf '%s\n' "$IMAGE_REF" > "$IMAGE_MARKER"
  printf '%s\n' "$RELEASE_PATH" > "$RELEASE_MARKER"
  printf '%s\n' "$COMPOSE_FILE" > "$COMPOSE_MARKER"
  ln -sfn "$RELEASE_PATH" "$CURRENT_LINK"
  compose "$IMAGE_REF" "$COMPOSE_FILE" ps
  docker image prune --force >/dev/null
  log "Deployment completed successfully"
  printf 'DEPLOYMENT_SUCCEEDED\n'
  exit 0
fi

log "Health check failed"
compose "$IMAGE_REF" "$COMPOSE_FILE" logs --tail 100 app || true

rollback || true

exit 1
