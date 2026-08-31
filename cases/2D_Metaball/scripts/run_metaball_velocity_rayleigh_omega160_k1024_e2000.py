"""Launch the audited omega=160 strict one-pass velocity-Rayleigh campaign."""


import run_metaball_velocity_rayleigh_k1024_e2000 as campaign


campaign.CONFIG = campaign.CASE_ROOT / (
    "configs/"
    "k1024_metaball_connected_siren_velocity_rayleigh_omega160_onepass_2000.json"
)
campaign.RESULT_ROOT = campaign.CASE_ROOT / (
    "results/siren_k1024_connected_velocity_rayleigh_omega160_onepass_e2000"
)
campaign.RUN_DIR = campaign.RESULT_ROOT / "run_001"
campaign.INPUTS = campaign.RESULT_ROOT / "inputs"
campaign.LAUNCH_MANIFEST = campaign.RESULT_ROOT / "launch_manifest.json"


if __name__ == "__main__":
    campaign.main()
