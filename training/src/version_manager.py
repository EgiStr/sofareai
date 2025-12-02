"""
Model Version Manager for SOFARE-AI.

Implements semantic versioning for ML models with:
- Version tracking (MAJOR.MINOR.PATCH)
- Release management
- Rollback capabilities
- MLflow integration
"""

import os
import re
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VersionBump(Enum):
    """Types of version increments."""
    MAJOR = "major"  # Breaking changes, architecture changes
    MINOR = "minor"  # New features, significant improvements
    PATCH = "patch"  # Bug fixes, minor improvements


class ReleaseStatus(Enum):
    """Status of a model release."""
    DRAFT = "draft"
    CANDIDATE = "candidate"
    RELEASED = "released"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """Represents a semantic version."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    @classmethod
    def parse(cls, version_string: str) -> "ModelVersion":
        """Parse a version string into ModelVersion."""
        # Pattern: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, version_string)
        
        if not match:
            raise ValueError(f"Invalid version string: {version_string}")
        
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5)
        )
    
    def bump(self, bump_type: VersionBump) -> "ModelVersion":
        """Create a new version with the specified bump."""
        if bump_type == VersionBump.MAJOR:
            return ModelVersion(self.major + 1, 0, 0)
        elif bump_type == VersionBump.MINOR:
            return ModelVersion(self.major, self.minor + 1, 0)
        else:  # PATCH
            return ModelVersion(self.major, self.minor, self.patch + 1)
    
    def __lt__(self, other: "ModelVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: "ModelVersion") -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))


@dataclass
class ModelRelease:
    """Represents a model release."""
    version: ModelVersion
    model_id: str
    mlflow_run_id: str
    status: ReleaseStatus
    created_at: datetime
    released_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    
    # Model metadata
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Artifacts
    artifacts_path: Optional[str] = None
    model_hash: Optional[str] = None
    
    # Release notes
    changelog: str = ""
    breaking_changes: List[str] = field(default_factory=list)
    deprecations: List[str] = field(default_factory=list)
    
    # Production status
    is_production: bool = False
    is_champion: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "version": str(self.version),
            "model_id": self.model_id,
            "mlflow_run_id": self.mlflow_run_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "artifacts_path": self.artifacts_path,
            "model_hash": self.model_hash,
            "changelog": self.changelog,
            "breaking_changes": self.breaking_changes,
            "deprecations": self.deprecations,
            "is_production": self.is_production,
            "is_champion": self.is_champion
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelRelease":
        return cls(
            version=ModelVersion.parse(data["version"]),
            model_id=data["model_id"],
            mlflow_run_id=data["mlflow_run_id"],
            status=ReleaseStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            released_at=datetime.fromisoformat(data["released_at"]) if data.get("released_at") else None,
            deprecated_at=datetime.fromisoformat(data["deprecated_at"]) if data.get("deprecated_at") else None,
            metrics=data.get("metrics", {}),
            parameters=data.get("parameters", {}),
            artifacts_path=data.get("artifacts_path"),
            model_hash=data.get("model_hash"),
            changelog=data.get("changelog", ""),
            breaking_changes=data.get("breaking_changes", []),
            deprecations=data.get("deprecations", []),
            is_production=data.get("is_production", False),
            is_champion=data.get("is_champion", False)
        )


class ModelVersionManager:
    """Manages model versions and releases."""
    
    def __init__(
        self,
        model_name: str = "sofarem3",
        registry_path: str = "./model_registry",
        mlflow_tracking_uri: Optional[str] = None
    ):
        """
        Initialize version manager.
        
        Args:
            model_name: Name of the model
            registry_path: Path to store version registry
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.model_name = model_name
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.releases_file = self.registry_path / "releases.json"
        self.releases: Dict[str, ModelRelease] = {}
        
        self.mlflow_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        
        self._load_releases()
    
    def _load_releases(self) -> None:
        """Load releases from file."""
        if self.releases_file.exists():
            with open(self.releases_file, "r") as f:
                data = json.load(f)
                self.releases = {
                    k: ModelRelease.from_dict(v) 
                    for k, v in data.get("releases", {}).items()
                }
    
    def _save_releases(self) -> None:
        """Save releases to file."""
        data = {
            "model_name": self.model_name,
            "releases": {k: v.to_dict() for k, v in self.releases.items()}
        }
        with open(self.releases_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_latest_version(self) -> Optional[ModelVersion]:
        """Get the latest version."""
        if not self.releases:
            return None
        
        versions = [r.version for r in self.releases.values()]
        return max(versions)
    
    def get_production_version(self) -> Optional[ModelRelease]:
        """Get the current production release."""
        for release in self.releases.values():
            if release.is_production and release.status == ReleaseStatus.RELEASED:
                return release
        return None
    
    def get_champion_version(self) -> Optional[ModelRelease]:
        """Get the current champion release."""
        for release in self.releases.values():
            if release.is_champion:
                return release
        return None
    
    def create_release(
        self,
        mlflow_run_id: str,
        bump_type: VersionBump = VersionBump.PATCH,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        changelog: str = "",
        breaking_changes: Optional[List[str]] = None,
        prerelease: Optional[str] = None
    ) -> ModelRelease:
        """
        Create a new model release.
        
        Args:
            mlflow_run_id: MLflow run ID for the model
            bump_type: Type of version bump
            metrics: Model metrics
            parameters: Model parameters
            changelog: Release notes
            breaking_changes: List of breaking changes
            prerelease: Prerelease identifier (e.g., "alpha.1", "rc.1")
        
        Returns:
            Created ModelRelease
        """
        # Determine new version
        latest = self.get_latest_version()
        if latest:
            new_version = latest.bump(bump_type)
        else:
            new_version = ModelVersion(1, 0, 0)
        
        if prerelease:
            new_version.prerelease = prerelease
        
        # Create release
        release = ModelRelease(
            version=new_version,
            model_id=f"{self.model_name}-{new_version}",
            mlflow_run_id=mlflow_run_id,
            status=ReleaseStatus.DRAFT if prerelease else ReleaseStatus.CANDIDATE,
            created_at=datetime.utcnow(),
            metrics=metrics or {},
            parameters=parameters or {},
            changelog=changelog,
            breaking_changes=breaking_changes or []
        )
        
        # Store artifacts
        release.artifacts_path = str(self.registry_path / "artifacts" / str(new_version))
        
        # Calculate model hash
        release.model_hash = self._calculate_model_hash(mlflow_run_id)
        
        self.releases[str(new_version)] = release
        self._save_releases()
        
        logger.info(f"Created release: {release.model_id} (status: {release.status.value})")
        return release
    
    def _calculate_model_hash(self, mlflow_run_id: str) -> str:
        """Calculate hash of model artifacts."""
        # In production, this would hash the actual model files
        return hashlib.sha256(mlflow_run_id.encode()).hexdigest()[:12]
    
    def promote_to_candidate(self, version: str) -> ModelRelease:
        """Promote a draft release to candidate."""
        release = self.releases.get(version)
        if not release:
            raise ValueError(f"Version {version} not found")
        
        if release.status != ReleaseStatus.DRAFT:
            raise ValueError(f"Only draft releases can be promoted to candidate")
        
        release.status = ReleaseStatus.CANDIDATE
        self._save_releases()
        
        logger.info(f"Promoted {version} to candidate")
        return release
    
    def release_version(self, version: str) -> ModelRelease:
        """
        Release a candidate version.
        
        This makes the version available for production deployment.
        """
        release = self.releases.get(version)
        if not release:
            raise ValueError(f"Version {version} not found")
        
        if release.status not in [ReleaseStatus.CANDIDATE, ReleaseStatus.DRAFT]:
            raise ValueError(f"Only draft/candidate releases can be released")
        
        release.status = ReleaseStatus.RELEASED
        release.released_at = datetime.utcnow()
        self._save_releases()
        
        logger.info(f"Released version {version}")
        return release
    
    def deploy_to_production(self, version: str) -> ModelRelease:
        """
        Deploy a released version to production.
        
        This sets the version as the current production version.
        """
        release = self.releases.get(version)
        if not release:
            raise ValueError(f"Version {version} not found")
        
        if release.status != ReleaseStatus.RELEASED:
            raise ValueError(f"Only released versions can be deployed to production")
        
        # Remove production flag from current production version
        current_prod = self.get_production_version()
        if current_prod:
            current_prod.is_production = False
        
        release.is_production = True
        self._save_releases()
        
        logger.info(f"Deployed {version} to production")
        return release
    
    def set_champion(self, version: str) -> ModelRelease:
        """Set a version as the champion model."""
        release = self.releases.get(version)
        if not release:
            raise ValueError(f"Version {version} not found")
        
        # Remove champion flag from current champion
        current_champion = self.get_champion_version()
        if current_champion:
            current_champion.is_champion = False
        
        release.is_champion = True
        self._save_releases()
        
        logger.info(f"Set {version} as champion")
        return release
    
    def deprecate_version(self, version: str, reason: str = "") -> ModelRelease:
        """Deprecate a version."""
        release = self.releases.get(version)
        if not release:
            raise ValueError(f"Version {version} not found")
        
        release.status = ReleaseStatus.DEPRECATED
        release.deprecated_at = datetime.utcnow()
        if reason:
            release.deprecations.append(reason)
        
        if release.is_production:
            release.is_production = False
            logger.warning(f"Deprecated version {version} was in production!")
        
        self._save_releases()
        
        logger.info(f"Deprecated version {version}")
        return release
    
    def rollback(self, target_version: Optional[str] = None) -> ModelRelease:
        """
        Rollback to a previous version.
        
        Args:
            target_version: Version to rollback to. If None, rolls back to previous production version.
        
        Returns:
            The rollback target release
        """
        current_prod = self.get_production_version()
        
        if target_version:
            target = self.releases.get(target_version)
            if not target:
                raise ValueError(f"Target version {target_version} not found")
        else:
            # Find previous production version
            released_versions = [
                r for r in self.releases.values()
                if r.status == ReleaseStatus.RELEASED and r != current_prod
            ]
            if not released_versions:
                raise ValueError("No previous released version found for rollback")
            
            target = max(released_versions, key=lambda r: r.version)
        
        # Mark current production as rolled back
        if current_prod:
            current_prod.status = ReleaseStatus.ROLLED_BACK
            current_prod.is_production = False
            logger.warning(f"Rolled back from {current_prod.version}")
        
        # Deploy target version
        target.is_production = True
        self._save_releases()
        
        logger.info(f"Rolled back to {target.version}")
        return target
    
    def get_release_history(self, limit: int = 10) -> List[ModelRelease]:
        """Get release history sorted by version descending."""
        releases = sorted(
            self.releases.values(),
            key=lambda r: r.version,
            reverse=True
        )
        return releases[:limit]
    
    def get_rollback_candidates(self) -> List[ModelRelease]:
        """Get versions that can be rolled back to."""
        current_prod = self.get_production_version()
        candidates = []
        for r in self.releases.values():
            if r.status == ReleaseStatus.RELEASED and r != current_prod:
                if current_prod is None or r.version < current_prod.version:
                    candidates.append(r)
        return candidates
    
    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two versions."""
        r1 = self.releases.get(version1)
        r2 = self.releases.get(version2)
        
        if not r1 or not r2:
            raise ValueError("One or both versions not found")
        
        # Compare metrics
        metric_diff = {}
        all_metrics = set(r1.metrics.keys()) | set(r2.metrics.keys())
        for metric in all_metrics:
            v1 = r1.metrics.get(metric, 0)
            v2 = r2.metrics.get(metric, 0)
            if v1 > 0:
                diff_pct = (v2 - v1) / v1 * 100
            else:
                diff_pct = 0
            metric_diff[metric] = {
                "version1": v1,
                "version2": v2,
                "diff": v2 - v1,
                "diff_pct": diff_pct
            }
        
        return {
            "version1": str(r1.version),
            "version2": str(r2.version),
            "metric_comparison": metric_diff,
            "parameter_changes": self._diff_parameters(r1.parameters, r2.parameters),
            "breaking_changes": r2.breaking_changes if r2.version > r1.version else r1.breaking_changes
        }
    
    def _diff_parameters(self, p1: Dict, p2: Dict) -> Dict:
        """Find differences between parameters."""
        changes = {}
        all_keys = set(p1.keys()) | set(p2.keys())
        
        for key in all_keys:
            if key not in p1:
                changes[key] = {"change": "added", "value": p2[key]}
            elif key not in p2:
                changes[key] = {"change": "removed", "value": p1[key]}
            elif p1[key] != p2[key]:
                changes[key] = {"change": "modified", "from": p1[key], "to": p2[key]}
        
        return changes
    
    def generate_changelog(
        self,
        from_version: Optional[str] = None,
        to_version: Optional[str] = None
    ) -> str:
        """Generate changelog between versions."""
        releases = self.get_release_history()
        
        if from_version:
            from_v = ModelVersion.parse(from_version)
            releases = [r for r in releases if r.version > from_v]
        
        if to_version:
            to_v = ModelVersion.parse(to_version)
            releases = [r for r in releases if r.version <= to_v]
        
        changelog = f"# Changelog for {self.model_name}\n\n"
        
        for release in releases:
            changelog += f"## [{release.version}] - {release.created_at.strftime('%Y-%m-%d')}\n\n"
            
            if release.changelog:
                changelog += f"{release.changelog}\n\n"
            
            if release.breaking_changes:
                changelog += "### Breaking Changes\n"
                for change in release.breaking_changes:
                    changelog += f"- {change}\n"
                changelog += "\n"
            
            if release.metrics:
                changelog += "### Metrics\n"
                for metric, value in release.metrics.items():
                    changelog += f"- {metric}: {value:.4f}\n"
                changelog += "\n"
        
        return changelog


# MLflow integration
def create_release_from_mlflow(
    manager: ModelVersionManager,
    run_id: str,
    bump_type: VersionBump = VersionBump.PATCH,
    changelog: str = ""
) -> ModelRelease:
    """
    Create a release from an MLflow run.
    
    Args:
        manager: Version manager instance
        run_id: MLflow run ID
        bump_type: Type of version bump
        changelog: Release notes
    
    Returns:
        Created ModelRelease
    """
    import mlflow
    
    if manager.mlflow_uri:
        mlflow.set_tracking_uri(manager.mlflow_uri)
    
    # Get run details
    run = mlflow.get_run(run_id)
    
    metrics = {k: v for k, v in run.data.metrics.items()}
    parameters = {k: v for k, v in run.data.params.items()}
    
    # Determine bump type from run tags
    if run.data.tags.get("version_bump"):
        bump_type = VersionBump(run.data.tags["version_bump"])
    
    # Get breaking changes from tags
    breaking_changes = []
    if run.data.tags.get("breaking_changes"):
        breaking_changes = run.data.tags["breaking_changes"].split(";")
    
    return manager.create_release(
        mlflow_run_id=run_id,
        bump_type=bump_type,
        metrics=metrics,
        parameters=parameters,
        changelog=changelog or run.data.tags.get("changelog", ""),
        breaking_changes=breaking_changes
    )


def determine_version_bump(
    current_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    architecture_changed: bool = False,
    features_changed: bool = False
) -> VersionBump:
    """
    Determine appropriate version bump based on changes.
    
    Args:
        current_metrics: Metrics from current production model
        new_metrics: Metrics from new model
        architecture_changed: Whether model architecture changed
        features_changed: Whether input features changed
    
    Returns:
        Recommended VersionBump
    """
    # Major: Architecture or feature changes
    if architecture_changed or features_changed:
        return VersionBump.MAJOR
    
    # Minor: Significant metric improvements
    if current_metrics and new_metrics:
        accuracy_key = "accuracy"
        if accuracy_key in current_metrics and accuracy_key in new_metrics:
            improvement = (new_metrics[accuracy_key] - current_metrics[accuracy_key]) / current_metrics[accuracy_key]
            if improvement > 0.05:  # More than 5% improvement
                return VersionBump.MINOR
    
    # Patch: Minor improvements or fixes
    return VersionBump.PATCH
