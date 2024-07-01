@echo off
set learning_rate=1e-2 1e-3 1e-4
set alpha=0.9 0.8 0.5
set outlook_steps=3 12 6 
set batch_size=16 32 64 8
set gradient_steps=1 2 10 50
set epsilon_start=0.5 0.2
set noise_std=0.1 0.2 0.5
set target_policy_noise=0.1 0.5
set policy_delay=2 5 10
for %%a in (%learning_rate%) do (
    for %%b in (%alpha%) do (
        for %%c in (%outlook_steps%) do (
            for %%d in (%batch_size%) do (
                for %%e in (%gradient_steps%) do (
                    for %%f in (%epsilon_start%) do (
                        for %%g in (%noise_std%) do (
                            for %%h in (%target_policy_noise%) do (
                                for %%i in (%policy_delay%) do (
                                    python test_td3.py --device cuda --learning_rate %%a --alpha %%b --outlook_steps %%c --batch_size %%d --gradient_steps %%e --epsilon_start %%f --noise_std %%g --target_policy_noise %%h --policy_delay %%i
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)
