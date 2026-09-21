"""Offline conditional quadratic initialization from training loss gradients."""
import torch
from .attacks import PICNNAttacker


def initialize_conditional_shift(attacker: PICNNAttacker, batches, loss_fn,
                                 lam: float, cost_scale: float = 1., ridge: float = .01,
                                 damping: float = .25):
    """Regress full training loss gradients on allowed identity-path contexts.

    At a quadratic anchor q=2, the affine coefficient b(h)=-s/lambda E[g|h]
    produces displacement s/(2lambda) E[g|h]. Labels and futures may supervise
    the OFFLINE regression; only the attacker's usual context enters its map.
    No incoming/test batch is fitted. Ridge penalizes slopes, not intercept.
    """
    if lam <= 0 or cost_scale <= 0 or ridge <= 0 or not 0 < damping <= 1:
        raise ValueError('Initialization scales must be positive')
    device=next(attacker.parameters()).device
    hdim=attacker.context_dim
    gram=torch.zeros(hdim+1,hdim+1,device=device,dtype=torch.double)
    cross=torch.zeros(hdim+1,attacker.energy.affine.out_features,device=device,dtype=torch.double)
    nodes=examples=0
    for batch in batches:
        x=batch.states.detach().requires_grad_(True)
        with torch.enable_grad():
            gradient=torch.autograd.grad(loss_fn(x,batch).sum(),x)[0].detach()
        with torch.no_grad():
            state=x.new_zeros(len(x),hdim);prefix=[]
            for t in range(x.shape[1]):
                update=attacker.source_encoder(attacker.input_normalization(x[:,t]),state)
                state=torch.where(batch.mask[:,t,None],update,state);prefix.append(state)
            history=torch.zeros_like(state)
            for t in range(x.shape[1]):
                future=torch.zeros_like(state) if attacker.threat=='causal' else prefix[-1]
                h=attacker.context_network(torch.cat([prefix[t],future,history],dim=-1))
                valid=batch.mask[:,t]
                features=torch.cat([h[valid],h.new_ones(int(valid.sum()),1)],dim=-1).double()
                target=gradient[valid,t].double()
                gram.add_(features.T@features);cross.add_(features.T@target)
                nodes+=len(features)
                update=attacker.history_encoder(attacker.input_normalization(x[:,t]),history)
                history=torch.where(valid[:,None],update,history)
            examples+=len(x)
    if not nodes:raise ValueError('Empty initialization training population')
    regularizer=torch.eye(hdim+1,device=device,dtype=torch.double)*ridge*nodes
    regularizer[-1,-1]=0
    coefficients=torch.linalg.solve(gram+regularizer,cross).to(attacker.energy.affine.weight.dtype)
    with torch.no_grad():
        attacker.energy.affine.weight.copy_((-damping*cost_scale/lam)*coefficients[:-1].T)
        attacker.energy.affine.bias.copy_((-damping*cost_scale/lam)*coefficients[-1])
    return dict(initializer='training_conditional_gradient_ridge',examples=examples,nodes=nodes,
                ridge=ridge,damping=damping,lambda_penalty=lam,reference_access='training_labels_only',
                slope_norm=float(coefficients[:-1].norm()),intercept_norm=float(coefficients[-1].norm()))
