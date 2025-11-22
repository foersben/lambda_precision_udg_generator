# Testing GitHub Actions Locally with Act

This guide explains how to test the GitHub Actions workflows locally before pushing to GitHub.

## Prerequisites

### Install Act

**Linux:**
```bash
# Using curl
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Or using package manager (if available)
# Ubuntu/Debian
sudo apt-get install act

# Arch Linux
sudo pacman -S act
```

**macOS:**
```bash
brew install act
```

**Windows:**
```bash
# Using Chocolatey
choco install act-cli

# Or using Scoop
scoop install act
```

### Install Docker

Act requires Docker to run. Install Docker Desktop or Docker Engine:
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Docker Engine](https://docs.docker.com/engine/install/)

## Configuration

The project includes an `.actrc` file with default configuration:
- Uses `catthehacker/ubuntu:act-latest` image (closest to GitHub's runners)
- Verbose output enabled
- Source directory binding

## Running Workflows

### Run All Workflows
```bash
# Run all jobs in all workflows
act

# Run all jobs for a specific event
act push
act pull_request
```

### Run Specific Jobs

```bash
# Run only linting
act -j lint-and-format

# Run only type checking
act -j type-check

# Run tests
act -j test

# Run coverage
act -j test-with-coverage
```

### Run Specific Workflows

```bash
# Run the CI workflow
act -W .github/workflows/ci.yml

# Run specific workflow and job
act -W .github/workflows/ci.yml -j test
```

## Common Use Cases

### 1. Test Before Committing
```bash
# Run quick checks (lint + format)
act -j lint-and-format

# If that passes, run full tests
act -j test
```

### 2. Test Specific Python Version
```bash
# The matrix will run all versions, but you can filter
act -j test --matrix python-version:3.11
```

### 3. Debug Failing Jobs
```bash
# Run with extra verbosity
act -j test -v

# Keep container running after failure for inspection
act -j test --reuse
```

### 4. Dry Run
```bash
# See what would run without actually running
act -n

# List all jobs
act -l
```

## Tips & Tricks

### Speed Up Runs

1. **Use smaller images for quick checks:**
   ```bash
   act -j lint-and-format -P ubuntu-latest=node:16-buster-slim
   ```

2. **Cache dependencies:**
   Act respects GitHub Actions cache, but initial runs will be slower.

3. **Skip jobs you don't need:**
   ```bash
   # Only run linting and one test job
   act -j lint-and-format -j test
   ```

### Environment Variables

```bash
# Set custom env vars
act -j test --env MY_VAR=value

# Use env file
act -j test --env-file .env.test
```

### Secrets

Create `.secrets` file (gitignored):
```
CODECOV_TOKEN=your_token_here
PYPI_TOKEN=your_token_here
```

Then act will automatically load them.

### Artifacts

Act stores artifacts in `/tmp/artifacts` by default:
```bash
# Specify custom artifact directory
act -j test-with-coverage --artifact-server-path ./act-artifacts
```

## Limitations of Act

1. **Not 100% identical:** Act simulates GitHub Actions but isn't perfect
2. **Some actions may not work:** Especially those requiring GitHub API
3. **Resource intensive:** Runs in Docker containers
4. **Slower first run:** Needs to download images

## Troubleshooting

### Docker Permission Issues
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Then log out and back in
```

### Container Cleanup
```bash
# Remove old act containers
docker ps -a | grep act- | awk '{print $1}' | xargs docker rm

# Clean up images
docker images | grep act | awk '{print $3}' | xargs docker rmi
```

### Act Not Finding Workflows
```bash
# Ensure you're in the project root
cd /path/to/lambda_precision_udg_generator

# List available workflows
act -l
```

## Example Workflow

Here's a complete local testing workflow:

```bash
# 1. Make your changes
vim src/lambdaprecisionudggenerator/utils/json_utils.py

# 2. Quick format check
act -j lint-and-format

# 3. If formatting needed, fix it
ruff format src/

# 4. Run type check
act -j type-check

# 5. Run tests locally first (faster)
.venv/bin/pytest tests/

# 6. If local tests pass, run full CI
act -j test

# 7. Finally, run coverage
act -j test-with-coverage

# 8. If all pass, commit and push!
git add .
git commit -m "Your changes"
git push
```

## Integration with Development

### Pre-commit Hook

Add to `.git/hooks/pre-push`:
```bash
#!/bin/bash
echo "Running local CI checks with act..."
act -j lint-and-format

if [ $? -ne 0 ]; then
    echo "Linting failed! Fix issues before pushing."
    exit 1
fi

echo "All checks passed!"
```

Make it executable:
```bash
chmod +x .git/hooks/pre-push
```

## Resources

- [Act Documentation](https://github.com/nektos/act)
- [Act Runner Images](https://github.com/catthehacker/docker_images)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Quick Reference

| Command | Description |
|---------|-------------|
| `act` | Run all workflows |
| `act -l` | List all jobs |
| `act -n` | Dry run |
| `act -j <job>` | Run specific job |
| `act -W <workflow>` | Run specific workflow |
| `act -v` | Verbose output |
| `act --reuse` | Keep containers after run |
| `act --env KEY=value` | Set environment variable |
| `act --secret-file .secrets` | Load secrets |
