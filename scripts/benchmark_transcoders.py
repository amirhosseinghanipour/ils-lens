"""Measure the registered all-layer update workload before 100M-token training.

Benchmark weights are discarded. Timings exclude corpus audit/checkpoint I/O and
are a throughput projection, never a training result or a completion estimate.
"""
import argparse,time
from pathlib import Path
import torch,yaml
from ilslens.model import load_local
from ilslens.data import wikitext_sequences,SAMPLING_POLICY
from ilslens.transcoders.train import SkipTranscoder
from ilslens.runtime import synchronize
from ilslens.experiment import Experiment
from scripts.train_transcoders import capture_mlp


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--config',default='configs/transcoders.yaml')
    p.add_argument('--output',default='runs/transcoder-benchmark');p.add_argument('--device',default='cuda')
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--steps',type=int,default=10);a=p.parse_args()
    if a.warmup<1 or a.steps<2:raise ValueError('At least one warmup and two measured updates required')
    config=yaml.safe_load(Path(a.config).read_text())
    run=Experiment(a.output,dict(config=dict(estimated_seconds=600,estimated_disk_mb=10),training_config=config,
        warmup=a.warmup,steps=a.steps,device=a.device,sampling_policy=SAMPLING_POLICY,protocol='all-layer-update-v1'))
    model,tokenizer=load_local(a.device,min_free_gib=config.get('estimated_gpu_gib',6.));d=model.config.hidden_size
    torch.manual_seed(config['seed']);ids,rows=wikitext_sequences(a.dataset,tokenizer,'train',a.warmup+a.steps)
    tcs=[SkipTranscoder(d,config['width_multiplier']*d,config['top_k']).to(model.device) for _ in model.model.layers]
    optimizers=[torch.optim.Adam(tc.parameters(),lr=config['learning_rate']) for tc in tcs]
    active=[torch.zeros(tc.encoder.out_features,dtype=torch.bool,device=model.device) for tc in tcs]
    if model.device.type=='cuda':torch.cuda.reset_peak_memory_stats(model.device)
    times=[]
    with capture_mlp(model) as (inputs,outputs):
        for index,tokens in enumerate(ids):
            synchronize(model.device);start=time.perf_counter()
            with torch.no_grad():model.model(tokens[None].to(model.device),use_cache=False)
            for layer,(tc,optimizer) in enumerate(zip(tcs,optimizers)):
                optimizer.zero_grad(set_to_none=True);features=tc.features(inputs[layer])
                active[layer]|=(features.detach()!=0).any(dim=(0,1))
                loss=(tc.decoder(features)+tc.skip(inputs[layer])-outputs[layer]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite transcoder benchmark loss')
                loss.backward();optimizer.step()
            synchronize(model.device);elapsed=time.perf_counter()-start
            if index>=a.warmup:times.append(elapsed)
            run.event('benchmark-update',index=index,warmup=index<a.warmup,seconds=elapsed)
    mean=sum(times)/len(times);steps=config['token_presentations']//128
    run.result(dict(seconds_per_update=times,mean_seconds_per_update=mean,tokens_per_second=128/mean,
        projected_update_seconds=steps*mean,registered_token_budget=config['token_presentations'],
        layers=len(tcs),hidden_size=d,width=config['width_multiplier']*d,top_k=config['top_k'],
        transcoder_parameters=sum(p.numel() for tc in tcs for p in tc.parameters()),
        optimizer_tensor_bytes=sum(v.numel()*v.element_size() for opt in optimizers for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device) if model.device.type=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device) if model.device.type=='cuda' else None,
        execution_device=str(model.device),tf32=False,dtype='float32',source_rows=rows,
        limitation='WikiText timing proxy; excludes web/chat tokenization, deduplication, corpus audit, checkpoint writes and evaluation. Raw repeated timings are not confidence intervals. Benchmark weights are discarded.'))

if __name__=='__main__':main()
