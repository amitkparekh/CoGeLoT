from contextlib import suppress

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError


def _check_if_word_in_instance_transform_target(word: str, instance_transform: DictConfig) -> bool:
    is_true = word in instance_transform["_target_"].lower()
    with suppress(ConfigKeyError):
        is_true = is_true or any(
            word in transform["_target_"].lower() for transform in instance_transform["transforms"]
        )
    return is_true


def _get_training_instruction(wandb_model_run_id: str) -> str:
    trained_instruction = {
        "Given": "original",
        "8lkml12g": "original",
        "2df3mwfn": "paraphrases",
        "ftwoyjb1": "original",
    }
    return trained_instruction[wandb_model_run_id]


def _is_trained_on_shuffled_obj(wandb_model_run_id: str) -> bool:
    return wandb_model_run_id == "ftwoyjb1"


def _get_difficulty(config: DictConfig) -> str:
    return OmegaConf.select(config, "model.difficulty", default="easy")


def _is_gobbledygook(instance_transform: DictConfig) -> bool:
    return _check_if_word_in_instance_transform_target("gobbledygook", instance_transform)


def _is_gobbledygook_word(instance_transform: DictConfig) -> bool:
    return _check_if_word_in_instance_transform_target(
        "word", instance_transform
    ) and _is_gobbledygook(instance_transform)


def _is_gobbledygook_tokens(instance_transform: DictConfig) -> bool:
    return _check_if_word_in_instance_transform_target(
        "token", instance_transform
    ) and _is_gobbledygook(instance_transform)


def _is_textual(instance_transform: DictConfig) -> bool:
    return _check_if_word_in_instance_transform_target("textual", instance_transform)


def _is_paraphrase(instance_transform: DictConfig) -> bool:
    return _check_if_word_in_instance_transform_target("reword", instance_transform)


def _is_disable_prompt_text(config: DictConfig) -> bool:
    return OmegaConf.select(config, "model.disable_prompt_text", default=False)


def _is_disable_prompt_visual(config: DictConfig) -> bool:
    return OmegaConf.select(config, "model.disable_prompt_visual", default=False)


def _is_shuffle_obj(config: DictConfig) -> bool:
    return OmegaConf.select(config, "model.should_shuffle_obj_per_observations", default=False)


def _get_evaluation_instance_transform_column(instance_transform: DictConfig) -> str:
    parameters: list[str] = []  # noqa: WPS110

    if _is_textual(instance_transform):
        parameters.append("textual")
    if _is_gobbledygook_word(instance_transform):
        parameters.append("gobbledygook_word")
    if _is_gobbledygook_tokens(instance_transform):
        parameters.append("gobbledygook_tokens")
    if _is_paraphrase(instance_transform):
        parameters.append("reworded")

    if not parameters:
        return "noop"
    return "_".join(parameters)


def _get_evaluation_prompt_modality_column(config: DictConfig) -> str:
    if _is_disable_prompt_text(config) and _is_disable_prompt_visual(config):
        return "disable_both"
    elif _is_disable_prompt_text(config):  # noqa: RET505
        return "disable_text"
    elif _is_disable_prompt_visual(config):
        return "disable_visual"
    return "disable_none"


def update_eval_config(config: DictConfig) -> DictConfig:
    """Update the evaluation config with all the necessary details."""
    wandb_model_run_id = OmegaConf.select(config, "model.model.wandb_run_id", default=None)
    instance_transform = OmegaConf.select(config, "model.vima_instance_transform")

    evaluation_instance_transform_column = _get_evaluation_instance_transform_column(
        instance_transform
    )
    evaluation_prompt_modality_column = _get_evaluation_prompt_modality_column(config)

    OmegaConf.update(
        config,
        "evaluation_instance_transform",
        evaluation_instance_transform_column,
        force_add=True,
        merge=False,
    )
    OmegaConf.update(
        config,
        "evaluation_prompt_modality",
        evaluation_prompt_modality_column,
        force_add=True,
        merge=False,
    )
    OmegaConf.update(
        config, "model.difficulty", _get_difficulty(config), force_add=True, merge=False
    )
    OmegaConf.update(
        config,
        "model.should_shuffle_obj_per_observations",
        _is_shuffle_obj(config),
        force_add=True,
        merge=False,
    )
    OmegaConf.update(
        config,
        "training_data",
        _get_training_instruction(wandb_model_run_id or "Given"),
        force_add=True,
        merge=False,
    )
    OmegaConf.update(
        config,
        "training_object_shuffled",
        _is_trained_on_shuffled_obj(wandb_model_run_id),
        force_add=True,
        merge=False,
    )
    return config


def build_eval_run_name(config: DictConfig) -> str:
    """Build the run name for the evaluation run."""
    wandb_model_run_id = OmegaConf.select(config, "model.model.wandb_run_id", default=None)
    instance_transform = OmegaConf.select(config, "model.vima_instance_transform")

    run_name = (
        _get_training_instruction(wandb_model_run_id)[:4].capitalize()
        if wandb_model_run_id
        else "Given"
    )
    run_name += "Shuf" if _is_trained_on_shuffled_obj(wandb_model_run_id) else ""

    evaluation_instance_transform_column = _get_evaluation_instance_transform_column(
        instance_transform
    )
    evaluation_prompt_modality_column = _get_evaluation_prompt_modality_column(config)
    difficulty = _get_difficulty(config)

    eval_instance_transform_to_name = {
        "textual": "Text",
        "gobbledygook_word": "GDGWord",
        "gobbledygook_tokens": "GDGToken",
        "reworded": "Para",
        "textual_gobbledygook_word": "Text + GDGWord",
        "textual_gobbledygook_tokens": "Text + GDGToken",
        "noop": "",
    }
    eval_prompt_modality_to_name = {
        "disable_both": "No Prompt",
        "disable_text": "No Text",
        "disable_visual": "No VisRef",
        "disable_none": "",
    }
    extras = [
        eval_instance_transform_to_name[evaluation_instance_transform_column],
        eval_prompt_modality_to_name[evaluation_prompt_modality_column],
    ]
    if _is_shuffle_obj(config):
        extras.append("ShufObj")

    extras = [extra for extra in extras if extra]
    if extras:
        run_name = f"{run_name} - {' + '.join(extras)}"
    if difficulty != "easy":
        run_name = f"{run_name} [{difficulty.capitalize()}]"

    return run_name
