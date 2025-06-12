"""Environment configurations for Ray."""

from dataclasses import dataclass
from typing import Dict, Optional

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnvConfig:
    """Environment configurations for Ray."""

    # General configurations.
    hf_token: Optional[str] = None
    ray_override_job_runtime_env: str = "1"

    # Ray Data configurations.
    ray_data_default_wait_for_min_actors_s: int = 600

    # The number of LLM engine replicas to use.
    num_replicas: int = 1
    # The batch size. This represents the unit of fault tolerance.
    # Smaller batch size implies more fault tolerance but may
    # introduce more overhead. Batch size should at least be 16 to
    # avoid hanging.
    batch_size: int = 256

    def gen_ray_runtime_envs(self, engine_envs: Dict[str, str]) -> Dict[str, str]:
        """Generate Ray runtime environment variables."""
        envs = {k.upper(): str(v) for k, v in engine_envs.items()}

        for key in (
            "hf_token",
            "ray_override_job_runtime_env",
            "ray_data_default_wait_for_min_actors_s",
        ):
            if getattr(self, key) is not None:
                envs[key.upper()] = str(getattr(self, key))
        return envs
