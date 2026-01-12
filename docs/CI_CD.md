# CI/CD Pipeline Documentation

## Overview

This project uses GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD). The pipeline automates testing, building, and deployment processes.

## Pipeline Stages

### 1. Lint
- **Purpose**: Check code quality and formatting
- **Tools**: flake8, black
- **Runs on**: Every push and pull request
- **What it does**:
  - Checks for syntax errors
  - Validates code style
  - Ensures code formatting compliance

### 2. Test
- **Purpose**: Run automated tests
- **Tools**: pytest, pytest-cov
- **Runs on**: Every push and pull request
- **What it does**:
  - Runs all unit tests
  - Generates coverage reports
  - Uploads coverage to codecov

### 3. Build
- **Purpose**: Build Docker image
- **Runs on**: After lint and test pass
- **What it does**:
  - Builds Docker image
  - Tests that the image can be imported
  - Validates container functionality

### 4. Train Model
- **Purpose**: Train ML model (if data available)
- **Runs on**: Only on pushes to main branch
- **What it does**:
  - Checks if training data exists
  - Trains the model
  - Uploads model artifacts

### 5. Security Scan
- **Purpose**: Security vulnerability scanning
- **Tools**: Trivy
- **Runs on**: Every push and pull request
- **What it does**:
  - Scans for known vulnerabilities
  - Uploads results to GitHub Security

### 6. Deploy
- **Purpose**: Deploy to production
- **Runs on**: Only on pushes to main branch
- **What it does**:
  - Deploys application (customize based on your infrastructure)

## Workflow File

The CI/CD pipeline is defined in `.github/workflows/ci-cd.yml`.

## Manual Triggers

You can manually trigger workflows:
1. Go to the "Actions" tab in GitHub
2. Select the workflow
3. Click "Run workflow"

## Branch Protection

Recommended branch protection rules:
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Require pull request reviews

## Environment Variables

Set these in GitHub Secrets (Settings > Secrets):
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password
- `REGISTRY_URL`: Container registry URL (if using private registry)

## Customization

### Adding New Tests
1. Add test files to `tests/` directory
2. Tests will run automatically in the test stage

### Adding New Linters
1. Install the linter in the lint job
2. Add the command to run the linter

### Custom Deployment
Modify the `deploy` job in `.github/workflows/ci-cd.yml` to match your deployment infrastructure.

## Status Badges

Add these badges to your README:

```markdown
![CI/CD](https://github.com/yourusername/repo/workflows/CI/CD%20Pipeline/badge.svg)
![Tests](https://codecov.io/gh/yourusername/repo/branch/main/graph/badge.svg)
```

## Troubleshooting

### Pipeline Fails on Lint
- Run `black app/ ml_pipeline/` to format code
- Fix flake8 errors manually

### Pipeline Fails on Tests
- Run tests locally: `pytest tests/ -v`
- Check test output for specific failures

### Docker Build Fails
- Test build locally: `docker build -t test .`
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt

### Model Training Fails
- Ensure data file exists at `Scripts/data.csv`
- Check data file format
- Review training logs

## Best Practices

1. **Always test locally** before pushing
2. **Keep workflows fast** - optimize slow steps
3. **Use caching** for dependencies when possible
4. **Fail fast** - stop pipeline on critical errors
5. **Monitor pipeline** - check Actions tab regularly

