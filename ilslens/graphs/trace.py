"""Expand direct target edges into a bounded graph of transcoder features."""
from .evaluate import evaluate_target

def source_identity(edge,graph):
    name=edge['node'];coordinate=edge['index']
    if name=='embedding':
        _,tokens,width=graph.clean_embeddings.shape;kind='embedding';layer=None
    else:
        kind,layer=name.split(':');layer=int(layer)
        shape=graph.clean_features[layer].shape if kind=='feature' else graph.clean_errors[layer].shape
        _,tokens,width=shape
    flat_position,unit=divmod(coordinate,width);batch,position=divmod(flat_position,tokens)
    if batch:raise ValueError('Local graphs currently require one prompt')
    return dict(id=f'{name}:{position}:{unit}',kind=kind,layer=layer,position=position,coordinate=unit)

def trace_graph(graph,target,root_result=None,max_feature_targets=8,max_depth=2,parents_per_target=2,edges_per_kind=3):
    """Lens/logit reads are targets; only features/errors/embeddings are nodes.

    Root completeness uses all direct source contributions. Expanding selected
    feature parents is for graph structure, and never double-counts their incoming
    attribution in that root budget. Unexpanded features are explicitly listed.
    """
    if min(max_feature_targets,max_depth,parents_per_target)<0:raise ValueError('Negative graph budget')
    root_result=evaluate_target(graph,target,edges_per_kind) if root_result is None else root_result
    nodes={};edges=[];targets=[dict(id='read-target',kind=target['kind'],layer=target.get('layer'),position=target['position'])]
    queue=[('read-target',0,root_result)];expanded=set();enqueued=set();feature_results=[]
    while queue:
        destination,depth,result=queue.pop(0)
        if not result['all_edges_pass']:raise AssertionError('A traced node has an invalid frozen edge')
        candidates=[]
        for checked in result['edge_checks']:
            source=source_identity(checked,graph);nodes[source['id']]=source
            edges.append(dict(source=source['id'],destination=destination,destination_is_target=destination=='read-target',**checked))
            if source['kind']=='feature':candidates.append((abs(checked['predicted']),source))
        for _,source in sorted(candidates,key=lambda pair:pair[0],reverse=True)[:parents_per_target]:
            if depth>=max_depth or len(enqueued)>=max_feature_targets or source['id'] in enqueued:continue
            enqueued.add(source['id']);feature_target=dict(kind='feature',layer=source['layer'],position=source['position'],feature=source['coordinate'])
            incoming=evaluate_target(graph,feature_target,edges_per_kind);incoming.pop('contributions')
            expanded.add(source['id']);feature_results.append(dict(node_id=source['id'],depth=depth+1,**incoming))
            queue.append((source['id'],depth+1,incoming))
    return dict(nodes=list(nodes.values()),targets=targets,edges=edges,expanded_features=feature_results,
        unexpanded_features=[node['id'] for node in nodes.values() if node['kind']=='feature' and node['id'] not in expanded],
        all_checked_edges_pass=all(edge['passes'] for edge in edges),
        limits=dict(max_feature_targets=max_feature_targets,max_depth=max_depth,parents_per_target=parents_per_target),
        freeze='Direct-source affine graph; incoming feature targets use their clean active encoder row (fixed TopK support)',
        survival='Only RMS denominators frozen in original model; feature-target survival recomputes the full TopK encoder',
        budget='Root budget is all direct source attribution; expanded ancestors are not counted again')
