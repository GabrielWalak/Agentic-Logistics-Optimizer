# Security Quick Reference

## 🚫 NEVER DO

```bash
# ❌ DON'T commit .env
git add .env
git commit -m "Add config"

# ❌ DON'T hardcode token
GITHUB_TOKEN = "ghp_..." in code

# ❌ DON'T put token in Dockerfile
RUN export GITHUB_TOKEN=ghp_...

# ❌ DON'T share token via email/Slack
"Here's my token: ghp_..."

# ❌ DON'T use same token everywhere
prod, staging, local all same token
```

## ✅ ALWAYS DO

### Local Development
```bash
# ✅ Create .env from example
cp .env.example .env

# ✅ Add YOUR token
nano .env
# GITHUB_TOKEN=ghp_your_token_here

# ✅ Verify it's ignored
grep "^\.env" .gitignore  # Should output: .env

# ✅ Before pushing
git status  # Should NOT show .env
```

### AWS Deployment
```bash
# ✅ Create secure env file in HOME
cat > ~/.agentic.env << 'EOF'
GITHUB_TOKEN=ghp_your_token_here
GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com
GITHUB_MODEL=gpt-4o-mini
EOF

# ✅ Restrict permissions
chmod 600 ~/.agentic.env

# ✅ Verify (should show: -rw-------)
ls -la ~/.agentic.env

# ✅ Run Docker with this file
docker run -d --env-file ~/.agentic.env agentic-ai-logistics:latest

# ✅ Verify token is NOT in image
docker inspect agentic-app | grep GITHUB_TOKEN  # Should be empty
```

### GitHub Token Management
```bash
# ✅ Create token with minimal scope
https://github.com/settings/tokens
# Select: gist or repo (minimal needed)

# ✅ Set expiration (90 days)
# https://github.com/settings/tokens

# ✅ Rotate every 90 days
# 1. Generate new token
# 2. Update ~/.agentic.env
# 3. docker restart agentic-app

# ✅ Revoke if compromised
# https://github.com/settings/tokens
# Click Delete on compromised token
```

---

## 📋 Checklist Before Push to GitHub

```bash
# 1. Check for secrets in staged changes
git diff --cached | grep -i "token\|ghp_\|secret"
# Should show: (nothing)

# 2. Check git history
git log -p -- .env | grep GITHUB_TOKEN
# Should show: (nothing)

# 3. Verify .gitignore
grep "\.env" .gitignore
# Should show: .env

# 4. Final check
git status | grep ".env"
# Should show: (nothing)

# ✅ Safe to push!
git push
```

---

## 🆘 If Token Gets Compromised

### Immediate Actions (< 5 min)

```bash
# 1. Revoke token immediately
https://github.com/settings/tokens  # Delete compromised token

# 2. Generate new token
https://github.com/settings/tokens  # Create new token

# 3. Update on EC2
ssh -i key.pem ubuntu@IP
nano ~/.agentic.env  # Update GITHUB_TOKEN
docker restart agentic-app

# 4. Verify new token works
docker logs agentic-app  # Should show "GitHub Models connected"
```

### Investigation (Next 30 min)

```bash
# Check if exposed online
# Visit: https://github.com/security/advisories

# Monitor GitHub API logs
# Check if token was used from unexpected IPs
# github.com Settings → Applications → check access logs

# Check EC2 logs
docker logs --since 10m agentic-app
aws ec2 describe-instances --instance-ids i-xxx

# Monitor billing
# AWS Console → Billing
# Check for unusual API usage charges
```

---

## 🔍 Verify Security

### Check token is NOT in Docker image
```bash
docker inspect agentic-app:latest | grep GITHUB_TOKEN
# Output: (nothing) ✅
```

### Check token is NOT in container running processes
```bash
docker exec agentic-app ps aux | grep GITHUB_TOKEN
# Output: (nothing) ✅
```

### Check token is NOT in git history
```bash
git log --all --oneline -- .env
# Output: (nothing) ✅

# OR check for ghp_ pattern
git log -p | grep "ghp_"
# Output: (nothing) ✅
```

### Check file permissions on EC2
```bash
ls -la ~/.agentic.env
# Should show: -rw------- (600)
# NOT:        -rw-r--r-- (644)
```

---

## 📚 See Also

- Full guide: **[SECURITY.md](SECURITY.md)**
- AWS deployment: **[QUICK_START_AWS.md](QUICK_START_AWS.md)**
- Build instructions: **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)**

---

## ⏱️ Token Rotation Schedule

```
Today (Day 0):
  ✅ Generate token with 90-day expiration
  ✅ Store in ~/.agentic.env
  ✅ Update deployment

Day 60:
  ⏰ REMINDER: Token expires in 30 days
  
Day 80:
  ⏰ URGENT: Token expires in 10 days
  
Day 90:
  ❌ Token expired - API calls will fail!
  ✅ Generate new token
  ✅ Update ~/.agentic.env
  ✅ docker restart agentic-app
```

---

**Bottom line:** 
- Keep .env in .gitignore ✅
- Store token in ~/.agentic.env (600 perms) ✅
- Rotate every 90 days ✅
- Revoke immediately if compromised ✅
