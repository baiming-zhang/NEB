"""Launch the audited omega=224 strict one-pass K=2048 campaign."""


import run_metaball_velocity_rayleigh_omega160_k2048_e2000 as campaign


campaign.CONFIG = campaign.CASE_ROOT / (
    "configs/"
    "k2048_metaball_connected_siren_velocity_rayleigh_omega224_onepass_2000.json"
)
campaign.RESULT_ROOT = campaign.CASE_ROOT / (
    "results/siren_k2048_connected_velocity_rayleigh_omega224_onepass_e2000"
)
campaign.RUN_DIR = campaign.RESULT_ROOT / "run_001"
campaign.INPUTS = campaign.RESULT_ROOT / "inputs"
campaign.LAUNCH_MANIFEST = campaign.RESULT_ROOT / "launch_manifest.json"
campaign.BANDWIDTH_AUDIT = campaign.RESULT_ROOT / "initial_bandwidth_audit.json"
campaign.EPOCH500_EVALUATION = (
    campaign.RESULT_ROOT / "evaluation/epoch0500_three_unseen_k2048_128"
)


if __name__ == "__main__":
    campaign.main()
