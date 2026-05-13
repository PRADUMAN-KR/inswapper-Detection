import torch


DEFAULT_SCORE_FUSION_WEIGHTS = {
    "real_fake": 0.55,
    "inswapper": 0.30,
    "boundary": 0.15,
}


def fuse_output_scores(
    outputs: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    weights = weights or DEFAULT_SCORE_FUSION_WEIGHTS
    return (
        weights["real_fake"] * torch.sigmoid(outputs["real_fake"]).flatten()
        + weights["inswapper"] * torch.sigmoid(outputs["inswapper"]).flatten()
        + weights["boundary"] * torch.sigmoid(outputs["boundary"]).flatten()
    )
