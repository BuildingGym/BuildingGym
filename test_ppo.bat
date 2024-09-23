@echo off
set learning_rate=1e-3 7e-3 1e-2 5e-4
set alpha=0.99 0.9 0.8
set outlook_steps=6
set batch_size=6 12 64
set vf_coef=0.5 0.1 0.9
set total_epoch=200
set ent_coef=0 0.1 0.2
set gae_lambda=1 0.95 0.8
set gamma=0.9 0.99 0.8
for %%a in (%learning_rate%) do (
    for %%b in (%alpha%) do (
        for %%c in (%outlook_steps%) do (
            for %%d in (%batch_size%) do (
                for %%e in (%vf_coef%) do (
                    for %%f in (%total_epoch%) do (
                        for %%g in (%ent_coef%) do (
                            for %%h in (%gae_lambda%) do (
                                for %%i in (%gamma%) do (
                                    python test_ppo_ext_var.py --learning_rate %%a --alpha %%b --outlook_steps %%c --batch_size %%d --vf_coef %%e --total_epoch %%f --ent_coef %%g --gae_lambda %%h --gamma %%i
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)
