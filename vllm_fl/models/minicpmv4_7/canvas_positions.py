"""Use the supplied 4.7 canvas reference, including its decode delta convention."""
import torch
from vllm_fl.models.minicpmv4_7.hf.mrope_minicpmv4_7 import build_image_bounds, get_rope_index


def special_ids(tokenizer):
    return dict(im_start_id=tokenizer.convert_tokens_to_ids('<image>'),
                im_end_id=tokenizer.convert_tokens_to_ids('</image>'),
                slice_start_id=tokenizer.convert_tokens_to_ids('<slice>'),
                slice_end_id=tokenizer.convert_tokens_to_ids('</slice>'),
                newline_id=tokenizer.encode('\n',add_special_tokens=False)[0])


def build_positions(input_tokens, features, token_ids, mode):
    if not features:
        return torch.arange(len(input_tokens)).unsqueeze(0).expand(3,-1), 0
    target_sizes=[]
    for feature in sorted(features,key=lambda f:f.mm_position.offset):
        key='tgt_sizes' if feature.modality=='image' else 'video_tgt_sizes'
        if feature.data is None or key not in feature.data:
            raise ValueError(f'Missing patch grids for canvas MRoPE: {feature.modality}')
        target_sizes.append(feature.data[key].data.reshape(-1,2).to(device='cpu',dtype=torch.long))
    targets=torch.cat(target_sizes)
    ids=torch.tensor(input_tokens,dtype=torch.long).unsqueeze(0)
    bounds=build_image_bounds(ids[0],token_ids)
    if bounds.shape[0]!=targets.shape[0]:
        raise ValueError(f'Visual boundaries/grids mismatch: {bounds.shape[0]} != {targets.shape[0]}')
    positions,deltas=get_rope_index(ids,None,[bounds],[targets],token_ids,mrope_mode=mode)
    return positions[:,0],int(deltas[0,0])
