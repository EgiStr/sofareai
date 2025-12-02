# Rollback Strategies

Guide to safely rolling back models and deployments in SOFARE-AI.

## Overview

Rollback is a critical capability for production ML systems. SOFARE-AI provides multiple rollback strategies:

- **Model Rollback**: Revert to a previous model version
- **Deployment Rollback**: Revert to a previous deployment state
- **Automatic Rollback**: Triggered by monitoring alerts

## Model Version Rollback

### Using Version Manager

```python
from training.src.version_manager import ModelVersionManager

# Initialize manager
manager = ModelVersionManager(
    model_name="sofarem3",
    registry_path="./model_registry"
)

# Get current production version
current = manager.get_production_version()
print(f"Current production: {current.version}")

# Get rollback candidates
candidates = manager.get_rollback_candidates()
for c in candidates:
    print(f"  - {c.version}: {c.status.value}")

# Rollback to previous version
rolled_back = manager.rollback()
print(f"Rolled back to: {rolled_back.version}")

# Or rollback to specific version
rolled_back = manager.rollback(target_version="1.2.0")
```

### CLI Commands

```bash
# List available versions
python -m training.src.version_manager list

# Show current production
python -m training.src.version_manager current

# Rollback to previous
python -m training.src.version_manager rollback

# Rollback to specific version
python -m training.src.version_manager rollback --version 1.2.0
```

## Deployment Rollback

### Docker Compose

```bash
# Quick rollback to previous image
docker compose down
docker compose pull  # Pull previous tagged images
docker compose up -d

# Or use specific version
export IMAGE_TAG=v1.2.0
docker compose up -d
```

### Kubernetes

```bash
# View rollout history
kubectl rollout history deployment/sofareai-serving

# Rollback to previous revision
kubectl rollout undo deployment/sofareai-serving

# Rollback to specific revision
kubectl rollout undo deployment/sofareai-serving --to-revision=3

# Check rollback status
kubectl rollout status deployment/sofareai-serving
```

### Blue-Green Rollback

```bash
# Switch traffic back to Blue environment
kubectl patch service sofareai \
  -p '{"spec":{"selector":{"version":"blue"}}}'

# Verify traffic switch
kubectl get endpoints sofareai
```

### Canary Rollback

```bash
# Route 100% traffic back to stable
kubectl patch virtualservice sofareai \
  --type merge \
  -p '{"spec":{"http":[{"route":[{"destination":{"host":"stable"},"weight":100}]}]}}'

# Scale down canary
kubectl scale deployment/sofareai-canary --replicas=0
```

## Automatic Rollback

### CI/CD Pipeline

The deployment pipeline includes automatic rollback triggers:

```yaml title=".github/workflows/ci-cd.yml"
deploy-production:
  steps:
    - name: Deploy Canary
      id: deploy
      run: ./scripts/deploy-canary.sh ${{ github.sha }}
    
    - name: Monitor Deployment
      id: monitor
      run: |
        # Check error rate
        sleep 300
        ERROR_RATE=$(curl -s prometheus/api/v1/query?query=rate(http_errors[5m]))
        
        if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
          echo "rollback=true" >> $GITHUB_OUTPUT
        fi
    
    - name: Rollback if needed
      if: steps.monitor.outputs.rollback == 'true'
      run: |
        echo "⚠️ Error rate too high, initiating rollback"
        ./scripts/rollback.sh
        exit 1
```

### Monitoring-Based Triggers

```python
from training.src.version_manager import ModelVersionManager
from training.src.drift_detector import DriftDetector

def check_and_rollback():
    """Check model health and trigger rollback if needed."""
    
    manager = ModelVersionManager()
    detector = DriftDetector()
    
    # Check for critical drift
    drift_result = detector.detect_multivariate_drift(current_data)
    
    if drift_result.severity == DriftSeverity.CRITICAL:
        logger.warning("Critical drift detected, initiating rollback")
        manager.rollback()
        notify_team("Model rolled back due to critical drift")
        return True
    
    # Check performance degradation
    current_metrics = get_current_metrics()
    baseline_metrics = get_baseline_metrics()
    
    perf_result = detector.detect_performance_degradation(
        baseline_metrics,
        current_metrics,
        degradation_threshold=0.1
    )
    
    if perf_result.drift_detected:
        logger.warning("Performance degradation detected, initiating rollback")
        manager.rollback()
        notify_team("Model rolled back due to performance degradation")
        return True
    
    return False
```

### Health Check Based Rollback

```yaml title="k8s/rollback-policy.yaml"
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: sofareai-serving
spec:
  replicas: 5
  strategy:
    canary:
      steps:
        - setWeight: 10
        - pause: {duration: 5m}
        - analysis:
            templates:
              - templateName: success-rate
        - setWeight: 50
        - pause: {duration: 10m}
        - analysis:
            templates:
              - templateName: success-rate
        - setWeight: 100
      
      # Automatic rollback on failure
      autoPromotionEnabled: false
      abortScaleDownDelaySeconds: 30
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  metrics:
    - name: success-rate
      interval: 1m
      successCondition: result[0] >= 0.99
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus:9090
          query: |
            sum(rate(http_requests_total{status!~"5.."}[5m]))
            /
            sum(rate(http_requests_total[5m]))
```

## Rollback Procedures

### Emergency Rollback

For critical production issues:

```bash
#!/bin/bash
# scripts/emergency-rollback.sh

echo "🚨 EMERGENCY ROLLBACK INITIATED"

# 1. Switch traffic immediately
kubectl patch virtualservice sofareai \
  --type merge \
  -p '{"spec":{"http":[{"route":[{"destination":{"host":"stable"},"weight":100}]}]}}'

# 2. Scale down problematic deployment
kubectl scale deployment/sofareai-canary --replicas=0

# 3. Verify stable is healthy
if ! curl -sf http://sofareai/health; then
  echo "❌ Stable environment unhealthy!"
  exit 1
fi

# 4. Mark version as rolled back
python -m training.src.version_manager deprecate $(cat .current-version)

# 5. Notify team
curl -X POST "$SLACK_WEBHOOK" \
  -d '{"text":"🚨 Emergency rollback completed"}'

echo "✅ Rollback completed"
```

### Planned Rollback

For scheduled maintenance or issues discovered post-deployment:

```bash
#!/bin/bash
# scripts/planned-rollback.sh

VERSION=${1:-"previous"}

echo "📋 Planned rollback to: $VERSION"

# 1. Get target version
if [ "$VERSION" == "previous" ]; then
  TARGET=$(python -m training.src.version_manager get-previous)
else
  TARGET=$VERSION
fi

echo "Target version: $TARGET"

# 2. Gradual traffic shift (reverse canary)
for weight in 90 50 10 0; do
  echo "Current deployment weight: $weight%"
  kubectl patch virtualservice sofareai \
    --type merge \
    -p "{\"spec\":{\"http\":[{\"route\":[{\"destination\":{\"host\":\"current\"},\"weight\":$weight},{\"destination\":{\"host\":\"rollback\"},\"weight\":$((100-weight))}]}]}}"
  
  sleep 60
  
  # Verify health
  if ! ./scripts/health-check.sh; then
    echo "❌ Health check failed during rollback"
    exit 1
  fi
done

# 3. Update version registry
python -m training.src.version_manager rollback --version $TARGET

echo "✅ Planned rollback completed"
```

## Post-Rollback Actions

### Incident Documentation

After any rollback:

```markdown
## Incident Report: Model Rollback

**Date**: 2024-01-15 14:30 UTC
**Duration**: 25 minutes
**Impact**: Predictions accuracy degraded

### Timeline
- 14:30: Canary deployment of v1.3.0 initiated
- 14:35: Error rate spike detected (5% → 12%)
- 14:40: Automatic rollback triggered
- 14:45: Traffic restored to v1.2.0
- 14:55: All systems stable

### Root Cause
Feature preprocessing bug in v1.3.0 caused incorrect normalization for high-volume periods.

### Resolution
- Rolled back to v1.2.0
- Bug fix in PR #234
- Additional test coverage added

### Action Items
- [ ] Add volume-based test cases
- [ ] Improve canary monitoring thresholds
- [ ] Review preprocessing pipeline
```

### Investigation Checklist

- [ ] Collect logs from failed deployment
- [ ] Export metrics during incident window
- [ ] Review model predictions during incident
- [ ] Check for data quality issues
- [ ] Analyze feature distributions
- [ ] Review recent code changes

## Rollback Testing

### Regular Rollback Drills

Schedule monthly rollback drills:

```python
def rollback_drill():
    """Execute planned rollback drill."""
    
    print("🎯 Starting rollback drill")
    
    # 1. Record current state
    current_version = manager.get_production_version()
    current_metrics = get_current_metrics()
    
    # 2. Execute rollback to previous version
    start_time = time.time()
    rolled_back = manager.rollback()
    rollback_duration = time.time() - start_time
    
    # 3. Verify rollback succeeded
    new_current = manager.get_production_version()
    assert new_current.version < current_version.version
    
    # 4. Check system health
    health_ok = check_system_health()
    assert health_ok
    
    # 5. Restore original version
    manager.deploy_to_production(str(current_version.version))
    
    # 6. Report results
    report = {
        "drill_date": datetime.now().isoformat(),
        "rollback_duration_seconds": rollback_duration,
        "from_version": str(current_version.version),
        "to_version": str(rolled_back.version),
        "health_check_passed": health_ok,
        "restored_successfully": True
    }
    
    print(f"✅ Drill completed in {rollback_duration:.1f}s")
    return report
```

## Best Practices

!!! tip "Version Management"
    
    - Always tag releases with semantic versions
    - Keep at least 3 previous versions available for rollback
    - Document breaking changes in each release
    - Test rollback procedures regularly

!!! tip "Monitoring"
    
    - Set up alerts for key metrics before deployment
    - Monitor error rates, latency, and prediction drift
    - Have clear rollback triggers defined
    - Keep dashboards ready for incident response

!!! warning "Common Pitfalls"
    
    - Not testing rollback procedures
    - Database migrations that can't be reversed
    - Losing model artifacts after deployment
    - Inadequate monitoring during canary phase

## Related Documentation

- [Deployment Guide](deployment.md)
- [A/B Testing](ab-testing.md)
- [Monitoring](monitoring.md)
- [Model Versioning](../development/cicd.md)
