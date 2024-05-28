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
        "0nsnkaer": "paraphrases",
        "ftwoyjb1": "original",
        "wa2rtion": "original",
        "b2slg2rh": "paraphrases",
        "drmm3ugl": "original",
        "bhuja4vo": "original",
        "wn9jc5l8": "original",
        "efxugme9": "original",
        "ln4nrqhg": "original",
        "53afo878": "original",
        "xivdgqm0": "original",
        "uuee5jre": "original",
        "6fmcpjg4": "original",
    }
    return trained_instruction[wandb_model_run_id]


def _get_prompt_conditioning_style(wandb_model_run_id: str) -> str:
    dec_only = {"bhuja4vo", "wn9jc5l8", "53afo878", "efxugme9"}
    if wandb_model_run_id in dec_only:
        return "dec_only"
    return "xattn"


def _get_visual_encoder_style(wandb_model_run_id: str) -> str:
    patches = {"efxugme9", "ln4nrqhg", "53afo878", "xivdgqm0" "uuee5jre"}
    if wandb_model_run_id in patches:
        return "patches"
    return "obj_centric"


def _is_trained_on_shuffled_obj(wandb_model_run_id: str) -> bool:
    return wandb_model_run_id in {"ftwoyjb1", "0nsnkaer", "wn9jc5l8"}


def _is_trained_without_text(wandb_model_run_id: str) -> bool:
    return wandb_model_run_id in {"uuee5jre", "6fmcpjg4"}


def _is_14_action_tokens(wandb_model_run_id: str) -> bool:
    return wandb_model_run_id in {"wa2rtion", "b2slg2rh", "53afo878", "xivdgqm0"}


def _is_trained_with_null_action(wandb_model_run_id: str) -> bool:
    return wandb_model_run_id in {"drmm3ugl"}


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


def _is_instruction_transform(instance_transform: DictConfig) -> bool:
    return _check_if_word_in_instance_transform_target("instruction", instance_transform)


def _is_disable_prompt_text(config: DictConfig) -> bool:
    return OmegaConf.select(config, "model.disable_prompt_text", default=False)


def _is_disable_prompt_visual(config: DictConfig) -> bool:
    return OmegaConf.select(config, "model.disable_prompt_visual", default=False)


def _is_shuffle_obj(config: DictConfig) -> bool:
    return OmegaConf.select(config, "model.should_shuffle_obj_per_observations", default=False)


def _get_evaluation_instance_transform_column(instance_transform: DictConfig) -> str:
    parameters: list[str] = []

    if _is_textual(instance_transform):
        parameters.append("textual")
    if _is_gobbledygook_word(instance_transform):
        parameters.append("gobbledygook_word")
    if _is_gobbledygook_tokens(instance_transform):
        parameters.append("gobbledygook_tokens")
    if _is_paraphrase(instance_transform):
        parameters.append("reworded")
    if _is_instruction_transform(instance_transform):
        parameters.append("diff_instruction")

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
    wandb_model_run_id = OmegaConf.select(config, "model.model.wandb_run_id", default=None).strip()
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
    OmegaConf.update(
        config,
        "num_action_tokens",
        14 if _is_14_action_tokens(wandb_model_run_id) else 1,
        force_add=True,
        merge=False,
    )
    OmegaConf.update(
        config,
        "trained_with_null_action",
        _is_trained_with_null_action(wandb_model_run_id),
        force_add=True,
        merge=False,
    )
    OmegaConf.update(
        config,
        "prompt_conditioning_style",
        _get_prompt_conditioning_style(wandb_model_run_id),
        force_add=True,
        merge=False,
    )
    OmegaConf.update(
        config,
        "visual_encoder_style",
        _get_visual_encoder_style(wandb_model_run_id),
        force_add=True,
        merge=False,
    )
    OmegaConf.update(
        config,
        "trained_without_text",
        _is_trained_without_text(wandb_model_run_id),
        force_add=True,
        merge=False,
    )
    return config


def build_eval_run_name(config: DictConfig) -> str:
    """Build the run name for the evaluation run."""
    wandb_model_run_id = OmegaConf.select(config, "model.model.wandb_run_id", default=None).strip()
    instance_transform = OmegaConf.select(config, "model.vima_instance_transform")

    prompt_conditioning_style = _get_prompt_conditioning_style(wandb_model_run_id)
    visual_encoder_style = _get_visual_encoder_style(wandb_model_run_id)

    prompt_condition_to_name = {
        "dec_only": "D",
        "xattn": "X",
    }
    visual_encoder_to_name = {
        "obj_centric": "Obj",
        "patches": "Ptch",
    }

    run_name = f"{prompt_condition_to_name[prompt_conditioning_style]}+"
    run_name += "Only" if _is_trained_without_text(wandb_model_run_id) else ""
    run_name += visual_encoder_to_name[visual_encoder_style]
    run_name += "Shuf" if _is_trained_on_shuffled_obj(wandb_model_run_id) else ""
    run_name += "+14" if _is_14_action_tokens(wandb_model_run_id) else ""
    run_name += " / "
    run_name += (
        _get_training_instruction(wandb_model_run_id)[:4].capitalize()
        if wandb_model_run_id
        else "Given"
    )
    run_name += "w/Null" if _is_trained_with_null_action(wandb_model_run_id) else ""

    evaluation_instance_transform_column = _get_evaluation_instance_transform_column(
        instance_transform
    )
    evaluation_prompt_modality_column = _get_evaluation_prompt_modality_column(config)
    difficulty = _get_difficulty(config)

    eval_instance_transform_to_name = {
        "textual": "ObjText",
        "gobbledygook_word": "GDGWord",
        "gobbledygook_tokens": "GDGToken",
        "reworded": "Para",
        "textual_gobbledygook_word": "ObjText + GDGWord",
        "textual_gobbledygook_tokens": "ObjText + GDGToken",
        "noop": "",
        "diff_instruction": "DiffInstr",
    }
    eval_prompt_modality_to_name = {
        "disable_both": "No Prompt",
        "disable_text": "No Text",
        "disable_visual": "No VisRef",
        "disable_none": "",
    }
    difficulty_to_name = {
        "easy": "",
        "medium": "Med",
        "hard": "Hard",
        "distracting": "Dstr",
        "extreme": "Xtr",
        "extremely_distracting": "XD",
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
        run_name = f"{run_name} [{difficulty_to_name[difficulty]}]"

    return run_name
