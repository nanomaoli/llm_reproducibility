"""Pipeline for batch processing large-scale LLM workloads."""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import ray
from ray.data._internal.stats import DatasetStats
from ray.data.dataset import Dataset
from ray.util import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .engines import EngineInitializerBase, init_engine_from_config
from .env_config import EnvConfig
from .logging import get_logger
from .tokenizer import Detokenizer
from .workload import EvalWorkload

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = get_logger(__name__)


class Pipeline:
    """Pipeline for batch processing large-scale LLM workloads.

    Args:
        engine_initializer: An engine initializer to create and initialize an engine.
        workload: Workload instance.
        env_config: EnvConfig to provide environment configurations of Ray.
    """

    def __init__(
        self,
        engine_initializer: EngineInitializerBase,
        env_config: EnvConfig,
    ):
        self.engine_initializer = engine_initializer
        self.env_config = env_config
        self.num_replicas: int = self.env_config.num_replicas
        self.ds: Optional[Dataset] = None
        self.stats: Optional[DatasetStats] = None

        self.pgs: List["PlacementGroup"] = []

        if not ray.is_initialized():
            ray.init(runtime_env={"env_vars": self.env_vars})

    @classmethod
    def from_config(
        cls, engine_cfg: Union[Dict[str, Any], str], workload: EvalWorkload, **kwargs
    ):
        """Initialize the pipeline from a configuration file or dictionary.

        Args:
            engine_cfg: A config file (in YAML) or a config dict. It should include
                the following keys: "engine", backend engine to use; "model",
                model to use; "accelerator_type", the GPU type; "configs",
                the engine specific configurations.
            workload: Workload instance.
            **kwargs: environment configuration parameters. See `EnvConfig` for more details.
        """
        engine_initializer = init_engine_from_config(engine_cfg)
        env_config = EnvConfig(**kwargs)
        return cls(engine_initializer, workload, env_config)

    @property
    def env_vars(self) -> Dict[str, Any]:
        return self.env_config.gen_ray_runtime_envs(
            self.engine_initializer.ray_env_vars
        )

    def load(
        self,
        repartition_by_batch_size: bool = False,
    ) -> Dataset:
        """Use the given workload to load and process the dataset,
        and then tokenize the prompts if needed. The processed dataset
        will be repartitioned based on the number of replicas and batch size.

        Args:
            repartition_by_batch_size: Whether to repartition the dataset by the
                batch size for fault tolerance granularity. You should enable
                this when the dataset is not from parquet and checkpointing is
                disabled.

        Returns:
            The processed dataset.
        """
        ds, num_blocks = self.workload.get_preprocessed_dataset(
            self.env_config.batch_size,
            repartition_by_batch_size,
        )
        if num_blocks is not None and num_blocks < self.num_replicas:
            logger.warning(
                "The number of blocks (%d) is less than the number of replicas (%d). "
                "This may result in suboptimal performance.",
                num_blocks,
                self.num_replicas,
            )

        if self.workload.need_tokenize:
            # TODO: Figure out a better concurrency.
            # Now we simply assume each LLM replica could have 4 tokenizers.
            # This is a heuristic and may not be optimal.
            tokenizer_concurrency = self.num_replicas * 4
            ds = ds.map_batches(
                self.workload.tokenizer_cls,
                fn_constructor_kwargs=self.workload.tokenizer_constructor_kwargs(
                    self.engine_initializer.model
                ),
                zero_copy_batch=True,
                concurrency=(1, tokenizer_concurrency),
                batch_size=self.env_config.batch_size,
            )

        # If max tokens in prompt is not set in the workload and max_model_len is not set
        # in the engine, we need to materialize the dataset to get the maximum tokens in prompt.
        # This may hurt the overall throughput but may be memory efficient.
        if self.workload.max_tokens_in_prompt == -1:
            if self.engine_initializer.max_model_len is not None:
                max_tokens = self.workload.sampling_params.get("max_tokens", 0)
                max_tokens_in_prompt = (
                    self.engine_initializer.max_model_len - max_tokens
                )
                msg = f"Max Prompt Tokens (max_model_len - max_tokens): {max_tokens_in_prompt}"
            else:
                logger.info(
                    "Materializing dataset after tokenization to get max prompt tokens"
                )
                ds = ds.materialize()

                max_tokens_in_prompt = int(ds.max("num_text_tokens"))
                msg = f"Max Prompt Tokens (inferred): {max_tokens_in_prompt}"
            self.workload.max_tokens_in_prompt = max_tokens_in_prompt
        else:
            msg = f"Max Prompt Tokens (specified in wokrload): {self.workload.max_tokens_in_prompt}"

        logger.info(msg)
        self.ds = ds
        return ds

    def __call__(self, workload: EvalWorkload):
        self.workload: EvalWorkload = workload
        # Set the task to "embed" if sampling params are not given.
        self.task_type_str: str = (
            "auto" if self.workload.sampling_params is not None else "embed"
        )
        return self.run(eager=False)

    def run(
        self,
        dataset: Optional[Dataset] = None,
        output_path: Optional[str] = None,
        detokenize: bool = True,
        eager: bool = True,
        repartition_by_batch_size: bool = False,
    ) -> Optional[Dataset]:
        """Perform batch processing on the dataset with LLM engines.

        Args:
            dataset: The dataset to process. If None, we directly use the given workload
                to load and process the dataset.
            output_path: The output path to write the processed dataset to parquet. It can be
                a path to a S3 bucket, or a path to local disk (with local:// as the prefix). If None,
                the processed dataset will be materialized but not be written.
            detokenize: Whether to detokenize the generated text. Default is True.
            eager: Whether to run the pipeline eagerly. If True, the dataset will be materialized.
                If False, we skip the materialization step and return the dataset. If output_path is specified,
                the dataset will be written to files and therefore will be materialized
                regardless of the eager flag.
            repartition_by_batch_size: Whether to repartition the dataset by the
                batch size for fault tolerance granularity. You should enable
                this when the dataset is not from parquet and checkpointing is
                disabled.

        Returns:
            The processed dataset. If output_path is not None, the dataset will be None after writing.
        """
        if not eager and output_path is not None:
            logger.warning("Eager mode is enforced because output path is specified")
            eager = True

        # Expend output_path in case environment variable is used.
        if output_path is not None:
            output_path = os.path.expanduser(output_path)

        # Force skipping detokenizer if task is "embed".
        if self.task_type_str == "embed" and detokenize:
            logger.info("Detokenization is skipped because of embedding workload")
            detokenize = False

        ray_remote_args = {}
        if self.engine_initializer.accelerator_type:
            ray_remote_args["accelerator_type"] = (
                self.engine_initializer.accelerator_type
            )
        ray_remote_args.update({"runtime_env": {"env_vars": self.env_vars}})

        if dataset is not None:
            self.ds = dataset
        elif self.ds is None:
            self.load(repartition_by_batch_size)
        assert self.ds is not None

        num_gpus = self.engine_initializer.num_gpus
        if self.engine_initializer.use_ray_placement_group:
            # Specify the number of GPUs required per LLM instance.
            # Note: for TP>1, num_gpus has to be 0 - instead, we specify a placement group
            if self.engine_initializer.num_gpus > 1:

                def _scheduling_strategy_fn(
                    num_gpus_per_instance: int, accelerator_type: str
                ):
                    def _get_bundle() -> Dict[str, float]:
                        bundle: Dict[str, float] = {"GPU": 1, "CPU": 1}
                        if accelerator_type:
                            bundle[f"accelerator_type:{accelerator_type}"] = 0.001
                        return bundle

                    pg = ray.util.placement_group(
                        [_get_bundle()] * num_gpus_per_instance,
                        strategy="STRICT_PACK",
                    )
                    self.pgs.append(pg)
                    return dict(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            pg, placement_group_capture_child_tasks=True
                        )
                    )

                ray_remote_args.update(
                    _scheduling_strategy_fn(
                        self.engine_initializer.num_gpus,
                        self.engine_initializer.accelerator_type,
                    )
                )

        self.ds = self.ds.map_batches(
            self.engine_initializer.get_engine_cls(),
            fn_constructor_kwargs=self.engine_initializer.get_engine_constructor_args(
                self.workload
            ),
            zero_copy_batch=True,
            # The number of running actors.
            concurrency=self.env_config.num_replicas,
            # The number of running batches for an actor in Ray Core level.
            # The value may not be optimal when the batch size is too small,
            # but it should be good enough for batch size >= 64.
            max_concurrency=4,
            batch_size=self.env_config.batch_size,
            num_gpus=num_gpus,
            **ray_remote_args,
        )

        # Skip detokenization. Usually used for tuning, profiling, and embedding.
        if detokenize:
            self.ds = self.ds.map_batches(
                Detokenizer,
                fn_constructor_kwargs={"model": self.engine_initializer.model},
                zero_copy_batch=True,
                concurrency=(1, self.num_replicas),
                batch_size=self.env_config.batch_size,
            )

        if output_path is not None:
            # Dataset will become None after writing to parquet.
            self.ds = self.ds.write_parquet(output_path)
        elif eager:
            self.ds = self.ds.materialize()

        # If the dataset pipeline is executed due to eager mode, we can cleanup.
        if eager:
            self.cleanup()

        return self.ds

    def cleanup(self):
        for pg in self.pgs:
            remove_placement_group(pg)
        self.pgs.clear()
