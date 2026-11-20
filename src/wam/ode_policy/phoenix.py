"""Read causal Joint96 conditions for the four action ODE experiments."""
import torch

from wam.inference.frame_joint import infer_frame_joint_window, mix_frame_action
from wam.model.modules.codecs.video_latent_codec import encode_input_image_latents_tensor
from wam.model.modules.conditioning.frame_attention import build_frame_memory_attention_mask
from wam.model.modules.conditioning.video_rope import sequence_start_video_rope_time_ids
from wam.model.modules.memory.proprio_encoder import append_proprio_to_context


def read_current_query(model, query, obs):
    """One current RGB frame per camera, one action query and exactly one MoT pass."""
    if not model.joint96_three_view or model.video_expert.action_conditioned:
        raise ValueError("Single-query reading requires unconditioned-video Joint96")
    if obs["camera_is_pad"].any():
        raise ValueError("Kitchen ODE recordings require all three current cameras")
    device, dtype = model.device, model.torch_dtype
    with torch.no_grad():
        context, mask = obs["context"].to(device, dtype), obs["context_mask"].to(device)
        if model.proprio_encoder is not None:
            context, mask = append_proprio_to_context(
                model, context=context, context_mask=mask, proprio=obs["proprio"].to(device, dtype))
        latent = encode_input_image_latents_tensor(
            model, input_image=obs["input_image"].to(device, dtype), tiled=False,
            video_layout=obs["video_layout"], video_view_names=obs["video_view_names"],
            camera_is_pad=obs["camera_is_pad"])
        time = latent.new_zeros(1)
        video = model.video_expert.pre_dit(
            x=latent, timestep=time, context=context, context_mask=mask,
            action=None, fuse_vae_embedding_in_latents=True, video_layout=obs["video_layout"],
            video_view_names=obs["video_view_names"],
            frame_ids=sequence_start_video_rope_time_ids(1, 1, device=device))
        action = model.action_expert.pre_dit(
            action_tokens=latent.new_zeros(1, 1, model.action_expert.action_dim),
            timestep=time, context=context, context_mask=mask, embodiment_ids=obs["embodiment_ids"])
    # Insert at the hidden-token interface, retaining frozen position, time and text conditioning.
    action["tokens"] = query.to(device=device, dtype=dtype)
    video_valid = torch.ones(video["tokens"].shape[:2], dtype=torch.bool, device=device)
    query_valid = torch.ones((1, 1), dtype=torch.bool, device=device)
    length = video_valid.shape[1]
    attention = build_frame_memory_attention_mask(
        video_seq_len=length, action_seq_len=1, video_tokens_per_frame=length,
        num_condition_frames=1, num_anchor_condition_frames=0, device=device)
    return mix_frame_action(model, video, action, attention, video_valid, query_valid, True)["action"]


def sample_block(model, head, obs, steps, seed, *, tokens):
    return infer_frame_joint_window(
        model, prompt=None, **obs, num_frames=9, action_horizon=head.horizon,
        num_inference_steps=steps, seed=seed, decode_video=False, return_action_tokens=tokens)


def read_prior(model, head, obs, *, active, denoise_steps, seed):
    """Raw parent output, also used by the legacy calibration and token-cache tools."""
    from .architectures import architecture_type
    return architecture_type(head.family).read(model, head, obs, active, denoise_steps, seed)
