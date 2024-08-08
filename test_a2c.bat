@echo off
set learning_rate=1e-3 7e-3 1e-2 5e-4
set alpha=0.99 0.9 0.8
set outlook_steps=6 12
set batch_size=6 12
set vf_coef=0.5 0.1 0.9
set pol_coef=1 0.8 0.5
set max_grad_norm=50
set max_grad_norm=50
set max_grad_norm=50
for %%a in (%learning_rate%) do (
    for %%b in (%alpha%) do (
        for %%c in (%outlook_steps%) do (
            for %%d in (%batch_size%) do (
                for %%e in (%vf_coef%) do (
                    for %%f in (%pol_coef%) do (
                        for %%g in (%max_grad_norm%) do (
                            for %%h in (%max_grad_norm%) do (
                                for %%i in (%max_grad_norm%) do (
                                    python test_a2c.py --learning_rate %%a --alpha %%b --outlook_steps %%c --batch_size %%d --vf_coef %%e --pol_coef %%f --max_grad_norm %%g --max_grad_norm %%h --max_grad_norm %%i
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)
