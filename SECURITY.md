# Security Guide - Managing GitHub Token Safely

## 🔐 The Problem
If `.env` with `GITHUB_TOKEN` gets committed to Git:
- ❌ Token exposed publicly
- ❌ Attacker can use your token to abuse GitHub Models API
- ❌ Your rate limit gets exhausted
- ❌ GitHub will disable the token

## ✅ The Solution

### Part 1: Local Development (Already Done ✓)

Your setup is already secure:

```
✅ .env is in .gitignore
✅ .env.example has placeholder values
✅ .env was never committed to Git
```

### Part 2: AWS EC2 Deployment (Recommended Options)

#### **Option A: Environment Variables (Simplest)**

```bash
# On EC2 instance, set environment variable directly
export GITHUB_TOKEN="ghp_your_token_here"

# Then run Docker with -e flag
docker run -d \
  --name agentic-app \
  -e GITHUB_TOKEN="$GITHUB_TOKEN" \
  -e GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com \
  -e GITHUB_MODEL=gpt-4o-mini \
  agentic-ai-logistics:latest
```

**Pros:** Simple, no files needed
**Cons:** Token visible in process list briefly

#### **Option B: .env File (Recommended)**

```bash
# On EC2 instance (NEVER on GitHub)
cat > ~/.agentic.env << 'EOF'
GITHUB_TOKEN=ghp_your_token_here
GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com
GITHUB_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_MAX_TOKENS=1000
REDIS_ENABLED=false
EOF

# Set permissions (only user can read)
chmod 600 ~/.agentic.env

# Run Docker
docker run -d \
  --name agentic-app \
  --env-file ~/.agentic.env \
  agentic-ai-logistics:latest
```

**Pros:** Secure, easy to manage
**Cons:** Need to create file manually on EC2

#### **Option C: AWS Secrets Manager (Most Secure - But Paid)**

```bash
# Store token in AWS Secrets Manager
aws secretsmanager create-secret \
  --name agentic-ai/github-token \
  --secret-string "ghp_your_token_here"

# Retrieve and use
TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id agentic-ai/github-token \
  --query SecretString --output text)

docker run -d \
  --name agentic-app \
  -e GITHUB_TOKEN="$TOKEN" \
  agentic-ai-logistics:latest
```

**Pros:** Enterprise-grade, encrypted, auditable
**Cons:** Costs $0.40/secret/month

#### **Option D: AWS Systems Manager Parameter Store (Free!)**

```bash
# Store token (free tier, 10k parameters)
aws ssm put-parameter \
  --name /agentic-ai/github-token \
  --value "ghp_your_token_here" \
  --type SecureString

# Retrieve and use
TOKEN=$(aws ssm get-parameter \
  --name /agentic-ai/github-token \
  --with-decryption \
  --query 'Parameter.Value' --output text)

docker run -d \
  --name agentic-app \
  -e GITHUB_TOKEN="$TOKEN" \
  agentic-ai-logistics:latest
```

**Pros:** Free, encrypted, AWS-native
**Cons:** Requires IAM permissions on EC2

---

## 🛡️ Best Practice Setup

### Step 1: Create GitHub Token (Limited Scope)

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. **Scopes** - Select ONLY:
   - `gist` (minimal scope for AI models)
   - `repo` (if needed for private repos)
4. **Expiration**: 90 days (rotate quarterly)
5. Copy token immediately (can't see again)

### Step 2: On Your Local Machine

```bash
# Create .env (NEVER commit!)
cp .env.example .env
nano .env
# Paste token

# Verify it's in .gitignore
grep "^\.env" .gitignore  # Should output: .env

# Double-check it won't be committed
git status  # Should NOT show .env

# If you accidentally added it, remove it
git rm --cached .env
git commit -m "Remove .env (should never be committed)"
```

### Step 3: On AWS EC2

```bash
# SSH into instance
ssh -i key.pem ubuntu@IP

# Create secure env file (accessible only by you)
cat > ~/.agentic.env << 'EOF'
GITHUB_TOKEN=ghp_your_token_here
GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com
GITHUB_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_MAX_TOKENS=1000
REDIS_ENABLED=false
EOF

# Lock down permissions (700 = rwx------)
chmod 700 ~/.agentic.env

# Verify permissions
ls -la ~/.agentic.env
# Should show: -rwx------ ubuntu ubuntu

# Run Docker with this file
docker run -d \
  --name agentic-app \
  --env-file ~/.agentic.env \
  --restart unless-stopped \
  agentic-ai-logistics:latest

# Verify it's running
docker logs agentic-app
```

### Step 4: Rotate Token Periodically

```bash
# Every 90 days:
# 1. Generate new token on GitHub
# 2. Update on EC2:
nano ~/.agentic.env
# Edit GITHUB_TOKEN
# 3. Restart container:
docker restart agentic-app
# 4. Verify in logs:
docker logs agentic-app
```

---

## 🚨 Security Checklist

- [ ] `.env` is in `.gitignore`
- [ ] `.env` was NEVER committed to Git
- [ ] `.env.example` has placeholder values only
- [ ] GitHub token has minimal scopes (gist or repo)
- [ ] Token expiration is set (90 days)
- [ ] On EC2: `.env` file has 700 permissions
- [ ] On EC2: `.env` is in home directory (not repo)
- [ ] Docker logs show "GitHub Models connected" (not "Token not found")
- [ ] Rotate token every 90 days
- [ ] No token in Docker image (stored in .env only)

---

## 🔍 How to Check if Token is Exposed

### Check Local Git History
```bash
cd ~/agentic-ai
git log -p -- .env | grep GITHUB_TOKEN
# Should show nothing
```

### Check Remote (GitHub)
```bash
# If token was committed and pushed:
git push -f
# VERY DANGEROUS! Only do if token is already compromised

# Better: Use git-filter-repo to rewrite history
pip install git-filter-repo
git filter-repo --path .env --invert-paths
git push -f --all
```

### Check if Token is Exposed Online
```bash
# Use GitHub API to check token status
curl -H "Authorization: token ghp_your_token_here" \
  https://api.github.com/user

# If 200 OK: token is valid and working
# If 401: token is invalid/revoked
```

---

## 🆘 If Token Gets Compromised

1. **Immediately revoke token**:
   - https://github.com/settings/tokens
   - Delete compromised token

2. **Generate new token** with same scopes

3. **Update on EC2**:
   ```bash
   nano ~/.agentic.env
   # Update GITHUB_TOKEN
   docker restart agentic-app
   ```

4. **Monitor usage**:
   - Check GitHub API logs
   - Monitor EC2 for unauthorized activity

---

## 📋 Git Best Practices

```bash
# Before every push, verify no secrets:
git diff --cached | grep -i "token\|secret\|password"
# Should return nothing

# Use pre-commit hook (optional but recommended)
# File: .git/hooks/pre-commit
#!/bin/bash
if git diff --cached | grep -i "ghp_\|token\|password"; then
  echo "❌ Secrets detected in commit!"
  exit 1
fi
```

---

## 🎯 Recommended Option for Your Setup

**For AWS t3.micro deployment:**

✅ **Option B: .env File** (Best balance of security & simplicity)

```bash
# On EC2:
cat > ~/.agentic.env << 'EOF'
GITHUB_TOKEN=ghp_your_token_here
GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com
GITHUB_MODEL=gpt-4o-mini
EOF

chmod 600 ~/.agentic.env

docker run -d \
  --name agentic-app \
  --env-file ~/.agentic.env \
  --restart unless-stopped \
  agentic-ai-logistics:latest
```

Why?
- ✅ Simple to implement
- ✅ Secure (file in home dir, restricted perms)
- ✅ No AWS API calls needed
- ✅ Easy to rotate token
- ✅ No additional costs

---

## Summary

```
❌ NEVER DO:
  - Commit .env to Git
  - Hardcode token in code
  - Share token via email
  - Use same token across environments

✅ ALWAYS DO:
  - Keep .env in .gitignore
  - Use .env.example with placeholders
  - Store token in secure location (EC2: ~/.agentic.env)
  - Set restrictive file permissions (600)
  - Rotate token every 90 days
  - Use minimal scopes for token
  - Monitor token usage
```
