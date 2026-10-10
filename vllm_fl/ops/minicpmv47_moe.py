# Copyright (c) 2026 BAAI. All rights reserved.
"""Bridge the installed KleidiAI G32 entry's FP32 activation-container ABI."""
import torch
import os
from functools import wraps
FAST_ENABLED=True
FAST_CALLS=0
ROUTED_ENABLED=os.environ.get('MINICPM_MOE_ROUTED','1')=='1'
ROUTED_CALLS=0
PREFILL_ENABLED=os.environ.get('MINICPM_MOE_PREFILL','0')=='1'
PREFILL_CALLS=0
FUSED_ACTIVATION_ENABLED=os.environ.get('MINICPM_MOE_FUSED_ACTIVATION','0')=='1'
FUSED_ACTIVATION_CALLS=0
PREFILL_SCHEDULER=os.environ.get('MINICPM_MOE_PREFILL_SCHEDULER','static')
DIRECT_ENABLED=os.environ.get('MINICPM_DIRECT_MOE','1')=='1'
DIRECT_CALLS=0
DIRECT_LAYERS=0
DIRECT_SHARED_A8_ENABLED=os.environ.get('MINICPM_MOE_SHARED_A8_DECODE','0')=='1'
DIRECT_SHARED_A8_CALLS=0
DIRECT_FUSED_GATES_ENABLED=os.environ.get('MINICPM_MOE_FUSED_GATES','0')=='1'
DIRECT_FUSED_ACTIVATION_ENABLED=os.environ.get('MINICPM_DIRECT_MOE_FUSED_ACTIVATION','0')=='1'
DIRECT_FUSED_ACTIVATION_CALLS=0
DIRECT_FUSED_GATES_CALLS=0


def install_direct_model_moe(model):
    """Bind the MiniCPM M=1 MoE graph directly to the native routed op.

    This removes the generic MoERunner/module-dispatch stack from decode while
    retaining the original forward for prefill and unsupported configurations.
    The eligibility checks intentionally describe this checkpoint exactly.
    """
    global DIRECT_LAYERS
    if os.environ.get('MINICPM_FAST_MOE','0')!='1' or not DIRECT_ENABLED:
        return 0
    from vllm_fl.ops.arm_operator_backend import flaggems_ops
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    native=flaggems_ops("w4a8_g32")
    shared_a8_native=None
    if DIRECT_SHARED_A8_ENABLED:
        shared_a8_native=flaggems_ops("moe")
    installed=0
    for layer in model.language_model.model.layers:
        block=getattr(layer,'mlp',None)
        experts=getattr(block,'experts',None)
        routed=getattr(experts,'routed_experts',None)
        config=getattr(experts,'moe_config',None)
        if block is None or experts is None or routed is None or config is None:
            continue
        if getattr(block.forward,'_minicpm47_direct_moe',False):
            continue
        eligible=(getattr(block,'tp_size',None)==1
            and getattr(block,'ep_size',None)==1
            and not getattr(block,'is_sequence_parallel',True)
            and not getattr(block,'enable_eplb',True)
            and getattr(block,'shared_expert',None) is not None
            and getattr(block.shared_expert,'expert_gate',None) is
                getattr(block,'shared_expert_gate',None)
            and getattr(experts,'gate',None) is getattr(block,'gate',None)
            and getattr(experts,'routed_input_transform',None) is None
            and getattr(experts,'routed_output_transform',None) is None
            and getattr(experts,'routed_scaling_factor',None)==1
            and getattr(config,'hidden_dim',None)==2048
            and getattr(config,'intermediate_size_per_partition',None)==512
            and getattr(config,'experts_per_token',None)==8
            and getattr(config,'routing_method',None) in
                (RoutingMethodType.Renormalize,RoutingMethodType.RenormalizeNaive))
        if not eligible:
            continue
        original=block.forward
        @wraps(original)
        def direct_forward(hidden_states,_block=block,_original=original,_routed=routed):
            global DIRECT_CALLS,DIRECT_SHARED_A8_CALLS,DIRECT_FUSED_GATES_CALLS,DIRECT_FUSED_ACTIVATION_CALLS,FAST_CALLS
            if (not DIRECT_ENABLED or hidden_states.device.type!='cpu'
                or hidden_states.dtype not in (torch.float32,torch.bfloat16)
                or hidden_states.shape!=(1,2048)):
                return _original(hidden_states)
            # vLLM creates the KAI packed tensors after model construction.
            # Resolve them here so construction and weight loading keep their
            # normal ordering.  The fallback also covers alternate backends.
            w1=getattr(_routed,'w13_weight_packed',None)
            w2=getattr(_routed,'w2_weight_packed',None)
            if (not isinstance(w1,torch.Tensor) or not isinstance(w2,torch.Tensor)
                or w1.dim()!=2 or w2.dim()!=2
                or w1.shape[0]!=256 or w2.shape[0]!=256):
                return _original(hidden_states)
            if DIRECT_FUSED_GATES_ENABLED:
                # Both gates are unquantized BF16 GEMVs over the same input.
                # Concatenating their rows removes one dispatcher/GEMV call.
                # Build this only after vLLM has finished loading weights.
                fused=getattr(_block,'_minicpm_fused_gate_weight',None)
                router_weight=getattr(_block.gate,'weight',None)
                shared_weight=getattr(_block.shared_expert_gate,'weight',None)
                if (not isinstance(fused,torch.Tensor)
                    and isinstance(router_weight,torch.Tensor)
                    and isinstance(shared_weight,torch.Tensor)
                    and router_weight.shape==(256,2048)
                    and shared_weight.shape==(1,2048)):
                    fused=torch.cat((router_weight.detach(),shared_weight.detach()),dim=0).contiguous()
                    _block._minicpm_fused_gate_weight=fused
                if isinstance(fused,torch.Tensor) and fused.shape==(257,2048):
                    gates=torch.nn.functional.linear(hidden_states,fused)
                    router_logits=gates[:,:256]
                    shared_gate=torch.sigmoid(gates[:,256:])
                    shared_mlp=_block.shared_expert
                    gate_up,_=shared_mlp.gate_up_proj(hidden_states)
                    shared=shared_mlp.act_fn(gate_up)
                    shared,_=shared_mlp.down_proj(shared)
                    shared=shared_gate*shared
                    DIRECT_FUSED_GATES_CALLS+=1
                else:
                    router_logits,_=_block.gate(hidden_states)
                    shared=_block.shared_expert(hidden_states)
            else:
                router_logits,_=_block.gate(hidden_states)
                shared=_block.shared_expert(hidden_states)
            if DIRECT_SHARED_A8_ENABLED:
                routed_output=shared_a8_native.moe(
                    hidden_states,router_logits,w1,w2,2048,512,8,True)
                DIRECT_SHARED_A8_CALLS+=1
            else:
                function=native.moe_decode_fused if DIRECT_FUSED_ACTIVATION_ENABLED else native.moe_decode_routed
                routed_output=function(
                    hidden_states,router_logits,w1,w2,2048,512,8,True)
                if DIRECT_FUSED_ACTIVATION_ENABLED:DIRECT_FUSED_ACTIVATION_CALLS+=1
            DIRECT_CALLS+=1
            FAST_CALLS+=1
            return shared+routed_output
        direct_forward._minicpm47_direct_moe=True
        block.forward=direct_forward
        installed+=1
    DIRECT_LAYERS+=installed
    return installed


def install():
    """Bridge vLLM's packed G32 MoE ABI without copying operator arithmetic."""
    from vllm.model_executor.layers.fused_moe.experts.cpu_int4_moe import CPUExpertsInt4
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm_fl.ops.arm_operator_backend import flaggems_ops

    original = CPUExpertsInt4.apply
    if getattr(original, '_fl_arm_g32_moe', False):
        return False
    native = flaggems_ops('w4a8_g32')

    @wraps(original)
    def apply(self,hidden_states,w1,w2,router_logits,activation,global_num_experts,
              expert_map,a1q_scale,apply_router_weight_on_input,num_expert_group=None,
              e_score_correction_bias=None,routed_scaling_factor=None,topk_group=None):
        kwargs = dict(num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,topk_group=topk_group)
        group32 = self.quant_config.block_shape is not None and self.quant_config.block_shape[1] == 32
        eligible = (group32 and FAST_ENABLED and
            hidden_states.device.type == 'cpu' and hidden_states.ndim == 2 and
            hidden_states.dtype in (torch.float32,torch.bfloat16) and
            not apply_router_weight_on_input and self._activation_kind(activation) in (0,2) and
            expert_map is None and num_expert_group is None and topk_group is None and
            e_score_correction_bias is None and routed_scaling_factor in (None,1.) and
            not self.quant_config.is_per_act_token)
        decode = hidden_states.shape[0] == 1
        if not eligible or not decode and not PREFILL_ENABLED:
            # Torch's KAI G32 entry consumes an FP32 activation container;
            # its arithmetic still dynamically quantizes activations to A8.
            value = hidden_states.float() if group32 else hidden_states
            out = original(self,value,w1,w2,router_logits,activation,global_num_experts,
                expert_map,a1q_scale,apply_router_weight_on_input,**kwargs)
            return out.to(hidden_states.dtype)
        renormalize = self.moe_config.routing_method in (
            RoutingMethodType.Renormalize,RoutingMethodType.RenormalizeNaive)
        if decode:
            function = native.moe_decode_fused if FUSED_ACTIVATION_ENABLED else native.moe_decode_routed
        elif PREFILL_SCHEDULER == 'adaptive':
            function = native.moe_prefill_adaptive
        elif PREFILL_SCHEDULER == 'balanced':
            function = native.moe_prefill_balanced
        elif PREFILL_SCHEDULER == 'static':
            function = native.moe_prefill_fused if FUSED_ACTIVATION_ENABLED else native.moe_prefill_routed
        else:
            raise ValueError('MINICPM_MOE_PREFILL_SCHEDULER must be static, adaptive, or balanced')
        return function(hidden_states,router_logits,w1,w2,self.moe_config.hidden_dim,
            self.moe_config.intermediate_size_per_partition,self.moe_config.experts_per_token,renormalize)

    apply._fl_arm_g32_moe = True
    CPUExpertsInt4.apply = apply
    return True

