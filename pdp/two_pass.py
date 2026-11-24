"""P1 (docs/design/2026-11-22-toi-uu-2-luot): the two PDP passes with the backbone and encoder run once.

PDP infers in two passes: a first pass without prompts gives the query, the second pass uses the prompts. The
prompts only enter the self-attention of the decoder, so the backbone and the encoder see the same input in both
passes. With reuse_features, the second pass takes the backbone features and the encoder output of the first one
and only runs the decoder again. Eval mode only: in train mode the encoder dropout would draw new masks.
"""

from models.modeling_deformable_detr import BackboneFeatures


def prompted_outputs(model, pixel_values, pixel_mask, *, use_prompts, local_query, reuse_features=False,
                     backbone_features: BackboneFeatures | None = None, first_task_id=None, second_task_id=None):
    """Outputs of the prompted pass (or of the single prompt-free pass when use_prompts is off).

    first_task_id / second_task_id: the task_id given to each pass, as the caller did before P1.
    backbone_features: precomputed by model.model.extract_features (e.g. shared with the student).
    """
    if reuse_features:
        if model.training:
            raise RuntimeError('P1: reuse_features reuses the encoder output, which is only valid in eval mode')
        if backbone_features is None:
            backbone_features = model.model.extract_features(pixel_values, pixel_mask)
    shared = {} if backbone_features is None else {'backbone_features': backbone_features}
    first = model(pixel_values=pixel_values, pixel_mask=pixel_mask, train=False,
                  **task_kwargs(first_task_id), **shared)
    if not use_prompts:
        return first
    # A (--per_query_prompt): the model config says every query selects its own prompts, whatever the caller passes
    every_query = local_query or getattr(model.config, 'per_query_prompt', 0)
    query = first.last_hidden_state if every_query else first.last_hidden_state.mean(dim=1)
    second = {**shared, 'encoder_outputs': (first.encoder_last_hidden_state,)} if reuse_features else shared
    return model(pixel_values=pixel_values, pixel_mask=pixel_mask, query=query, train=False,
                 **task_kwargs(second_task_id), **second)


def task_kwargs(task_id):
    return {} if task_id is None else {'task_id': task_id}
