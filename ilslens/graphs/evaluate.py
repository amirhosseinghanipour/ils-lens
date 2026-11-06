"""Direct source-to-target edges, finite interventions, and attribution budgets."""
import torch
from ..hooks import capture_blocks

def target_value(graph,target,forward,frozen_gate=True):
    if target['kind']=='logit':
        logits=forward();value=logits[0,target['position'],target['token']]
        if target.get('contrast_token') is not None:value=value-logits[0,target['position'],target['contrast_token']]
        return value
    if target['kind']=='lens':
        with capture_blocks(graph.model,[target['layer']]) as states:forward()
        return states[target['layer']][0,target['position']]@target['vector']
    if target['kind']=='feature':
        layer=target['layer'];feature=target['feature'];position=target['position'];captured=[]
        handle=graph.model.model.layers[layer].mlp.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        try:forward()
        finally:handle.remove()
        h=captured[0][0,position];tc=graph.transcoders[layer]
        if not frozen_gate:return tc.features(h)[feature]
        if graph.clean_features[layer][0,position,feature]<=0:raise ValueError('Feature target must be active in the clean TopK support')
        return h@tc.encoder.weight[feature]+tc.encoder.bias[feature]
    raise ValueError('Unknown graph target')

def original_effect(graph,target,node_name,index,change):
    """A source-output patch in the original model with only norms frozen."""
    handle=None
    with graph.norms_only(),torch.no_grad():
        forward=lambda:graph.model(graph.ids,use_cache=False).logits
        clean=target_value(graph,target,forward,frozen_gate=False)
        if node_name=='embedding':
            def patch(module,args,out):
                result=out.clone();result.flatten()[index]+=change;return result
            handle=graph.model.model.embed_tokens.register_forward_hook(patch)
        else:
            kind,layer=node_name.split(':');layer=int(layer)
            node=graph.nodes[node_name];flat_position,coordinate=divmod(index,node.shape[-1])
            batch,position=divmod(flat_position,node.shape[1])
            displacement=graph.outputs[layer].new_zeros(graph.outputs[layer].shape[-1])
            if kind=='feature':displacement=graph.transcoders[layer].decoder.weight[:,coordinate]*change
            else:displacement[coordinate]=change
            def patch(module,args,out):
                result=out.clone();result[batch,position]+=displacement;return result
            handle=graph.model.model.layers[layer].mlp.register_forward_hook(patch)
        try:patched=target_value(graph,target,forward,frozen_gate=False)
        finally:handle.remove()
    return float(patched-clean)

def evaluate_target(graph,target,edges_per_kind=3,far_upstream_distance=4):
    checks=[]
    with graph.frozen():
        value=target_value(graph,target,graph.forward);attribution=graph.contributions(value)
        original_nodes=dict(graph.nodes)
        contributions=attribution['contributions'];target_layer=target.get('layer',len(graph.model.model.layers)-1)
        total=sum(float(v.abs().sum()) for v in contributions.values())+float(attribution['bias'].abs())
        far=sum(float(v.abs().sum()) for name,v in contributions.items()
                if name=='embedding' or int(name.split(':')[1])<=target_layer-far_upstream_distance)
        candidates=[]
        for kind in ('feature','error','embedding'):
            scores=[]
            for name,mass in contributions.items():
                if not name.startswith(kind):continue
                values,indices=mass.abs().flatten().topk(min(edges_per_kind,mass.numel()))
                scores.extend((float(v),name,int(i)) for v,i in zip(values,indices) if v>1e-9)
            candidates.extend(sorted(scores,reverse=True)[:edges_per_kind])
        for _,name,index in candidates:
            node=original_nodes[name]
            gradient=torch.autograd.grad(value,node,retain_graph=True)[0]
            change=-node.flatten()[index].detach().clone();predicted=float(gradient.flatten()[index]*change)
            if name=='embedding':
                saved=graph.clean_embeddings.flatten()[index].clone()
                graph.clean_embeddings.flatten()[index]+=change
            else:
                with torch.no_grad():node.flatten()[index]+=change
            with torch.no_grad():actual=float(target_value(graph,target,graph.forward)-value.detach())
            if name=='embedding':graph.clean_embeddings.flatten()[index]=saved
            else:
                with torch.no_grad():node.flatten()[index]-=change
            error=abs(actual-predicted)/max(abs(actual),abs(predicted),1e-7)
            checks.append(dict(node=name,index=index,change=float(change),predicted=predicted,frozen_effect=actual,
                               relative_error=error,passes=error<1e-3))
        completeness=attribution['completeness']
    # Keep source tensors for shape metadata; restore original attention and MLPs.
    from .intervention import replacement_effect
    for edge in checks:
        effect=original_effect(graph,target,edge['node'],edge['index'],edge['change'])
        edge['norm_only_effect']=effect
        edge['survival_ratio']=effect/edge['predicted'] if abs(edge['predicted'])>1e-7 else None
        name=edge['node'];index=edge['index'];change=edge['change']
        if name=='embedding':
            layer=None;displacement=torch.zeros_like(graph.clean_embeddings)
            displacement.flatten()[index]=change
        else:
            kind,layer=name.split(':');layer=int(layer)
            width=graph.clean_features[layer].shape[-1] if kind=='feature' else graph.clean_errors[layer].shape[-1]
            flat_position,coordinate=divmod(index,width)
            batch,position=divmod(flat_position,graph.ids.shape[1])
            displacement=torch.zeros_like(graph.outputs[layer])
            if kind=='feature':displacement[batch,position]=graph.transcoders[layer].decoder.weight[:,coordinate]*change
            else:displacement[batch,position,coordinate]=change
        live=replacement_effect(graph,target,layer,displacement)
        live['direct_edge_prediction']=edge['predicted']
        live['direct_edge_relative_error']=abs(live['intervention_effect']-edge['predicted'])/max(abs(live['intervention_effect']),abs(edge['predicted']),1e-7)
        edge['live_replacement']=live
    return dict(target_value=float(value.detach()),graph_completeness=completeness,far_upstream_share=far/total if total else None,
        far_upstream_distance=far_upstream_distance,edge_checks=checks,all_edges_pass=bool(checks) and all(e['passes'] for e in checks),
        contributions={name:value.cpu() for name,value in contributions.items()},bias=float(attribution['bias']),
        live_replacement_all_local_derivatives_pass=bool(checks) and all(e['live_replacement']['passes'] for e in checks),
        edge_definition='Direct source edges exclude paths through other feature activations; live_replacement reports the distinct propagated effect',
        norm_only_definition='Original attention and MLPs; clean RMS denominators frozen; decoded feature-output or error-output patch')
