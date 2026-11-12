// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Experimental typed CPU framework bridge. Arithmetic remains in stock vLLM;
// FlagGems' cached TLE Raw invocation surrounds validation, workspace and calls.
#include "raw_dispatch.h"
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <array>
#include <atomic>
#include <string>
#include <vector>
namespace {
std::atomic<uint64_t> calls{0};
struct Workspace {
    std::array<int64_t,4> shape{};
    at::Tensor buffer,key,value;
};
at::Tensor encoder_attention(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& slots,const at::Tensor& starts,const at::Tensor& lengths,
    const at::Tensor& blocks,const at::Tensor& schedule,double scale,const std::string& isa,
    int64_t cache_blocks,int64_t block_size,int64_t window,double cap) {
    return flag_gems::arm::invoke_raw([&] {
        TORCH_CHECK(q.dim()==3 && q.scalar_type()==at::kBFloat16 && q.is_cpu()
            && k.sizes()==q.sizes() && v.sizes()==q.sizes()
            && k.scalar_type()==q.scalar_type() && v.scalar_type()==q.scalar_type()
            && k.is_cpu() && v.is_cpu(),"Expected matching CPU BF16 encoder Q/K/V");
        const std::array<int64_t,4> shape{cache_blocks,q.size(1),block_size,2*q.size(2)};
        thread_local Workspace workspace;
        if(!workspace.buffer.defined() || workspace.shape!=shape) {
            auto buffer=at::zeros(shape,q.options());
            auto parts=buffer.view({cache_blocks,q.size(1),2*block_size,q.size(2)}).chunk(2,2);
            workspace.buffer=std::move(buffer);
            workspace.key=std::move(parts[0]);workspace.value=std::move(parts[1]);
            workspace.shape=shape;
        }
        auto output=at::empty(q.sizes(),q.options());
        static const auto cache=c10::Dispatcher::singleton().findSchemaOrThrow("_C::cpu_attn_reshape_and_cache","");
        static const auto attention=c10::Dispatcher::singleton().findSchemaOrThrow("_C::cpu_attention_with_kv_cache","");
        std::vector<c10::IValue> cache_args{k,v,workspace.key,workspace.value,slots,isa,1.0,1.0,std::string("auto")};
        cache.callBoxed(&cache_args);
        std::vector<c10::IValue> args{q,workspace.key,workspace.value,output,starts,lengths,scale,false,
            c10::IValue(),c10::SymInt(window),blocks,cap,schedule,c10::IValue(),c10::IValue(),1.0,1.0,std::string("auto")};
        attention.callBoxed(&args);
        calls.fetch_add(1,std::memory_order_relaxed);
        return output;
    });
}
// The stable C++ wrapper can dispatch this integer-only schema directly. The
// general string/float schema currently generates a Python custom-op callback.
at::Tensor encoder_attention_arm(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& slots,const at::Tensor& starts,const at::Tensor& lengths,
    const at::Tensor& blocks,const at::Tensor& schedule,
    int64_t cache_blocks,int64_t block_size,int64_t window) {
    TORCH_CHECK(q.dim()==3 && q.size(2)==64,"Arm attention entry requires head dimension64");
    return encoder_attention(q,k,v,slots,starts,lengths,blocks,schedule,
        0.125,"neon",cache_blocks,block_size,window,0.0);
}
}
extern "C" uint64_t vllm_fl_cpu_attention_raw_calls(){return calls.load(std::memory_order_relaxed);}
TORCH_LIBRARY_FRAGMENT(vllm_fl_cpu_native,m) {
    m.def("encoder_attention.arm(Tensor q, Tensor k, Tensor v, Tensor slots, Tensor starts, Tensor lengths, Tensor blocks, Tensor schedule, int cache_blocks, int block_size, int window) -> Tensor");
    m.def("encoder_attention.raw(Tensor q, Tensor k, Tensor v, Tensor slots, Tensor starts, Tensor lengths, Tensor blocks, Tensor schedule, float scale, str isa, int cache_blocks, int block_size, int window, float cap) -> Tensor");
}
TORCH_LIBRARY_IMPL(vllm_fl_cpu_native,CPU,m) {
    m.impl("encoder_attention.arm",encoder_attention_arm);
    m.impl("encoder_attention.raw",encoder_attention);
}
