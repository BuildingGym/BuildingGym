@echo off
set learning_rate=1e-3 5e-3 1e-2 5e-4
set alpha=0.99 0.9 0.8
set epsilon_start=0.5 0.2
set batch_size=256 512 128
set policy_delay=2 5 10
set max_buffer_size=2000 4000 10000 20000
set gamma=0.99 0.9 0.5
set gradient_steps=1 5
set target_policy_noise=0.2 0.1
for %%a in (%learning_rate%) do (
    for %%b in (%alpha%) do (
        for %%c in (%epsilon_start%) do (
            for %%d in (%batch_size%) do (
                for %%e in (%policy_delay%) do (
                    for %%f in (%max_buffer_size%) do (
                        for %%g in (%gamma%) do (
                            for %%h in (%gradient_steps%) do (
                                for %%i in (%target_policy_noise%) do (
                                    python test_td3.py --learning_rate %%a --alpha %%b --epsilon_start %%c --batch_size %%d --policy_delay %%e --max_buffer_size %%f --gamma %%g --gradient_steps %%h --target_policy_noise %%i
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)
