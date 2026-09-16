"""Run the same trainer with a single explicit gauge ablation switch."""
import train_gauge_canonical as trainer
from ablation_backend import _streamfunction_dirichlet_rayleigh_peel_step

original_validate=trainer.validate_config

def validate_config(config):
    enabled=config.get('gauge_fixing_enabled',True)
    if not isinstance(enabled,bool):
        raise ValueError('gauge_fixing_enabled must be a boolean')
    expected='mean_zero_before_sort_normalize_mgs' if enabled else 'none'
    if config.get('gauge_canonicalization')!=expected:
        raise ValueError('gauge switch and canonicalization metadata disagree')
    original_validate(dict(config,gauge_canonicalization='mean_zero_before_sort_normalize_mgs'))

def peel_step(cfg,model,points,beta,alpha,training,reference_model=None,full_diagnostics=True):
    if reference_model is not None:
        raise ValueError('No reference model in paired ablation')
    return _streamfunction_dirichlet_rayleigh_peel_step(cfg,model,points,beta,alpha,training,full_diagnostics)

trainer.validate_config=validate_config
trainer.peel_step=peel_step

if __name__=='__main__': trainer.main()
