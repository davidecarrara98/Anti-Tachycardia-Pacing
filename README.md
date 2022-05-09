# Homework 1 - Anti-tachycardia pacing

## :memo: Description

Anti-tachycardia pacing (ATP) delivers pacing pulses to interrupt a tachyarrhythmia episode and restore normal sinus rhythm. Devices such as the implantable cardioverter-defibrillator (ICD) employ this technique before delivering high-voltage shocks to reduce the patient's sensation of pain and extend the life of the device's battery.

To ensure that ATP is effective, these devices must be programmed by selecting the proper pacing pulse width (fixed in the simulations), duration and timing. Note that it is preferred to keep the impulse limited, to reduce battery consumption.

The goal of this homework is to identify an optimal ATP strategy based on a single impulse.
The dataset comprises a single bipolar trace (first row), the time discretization (second row), and an impulse shape for three different patients (third row).


This process can be modeled using the monodomain model coupled with a system of ODE given by the Rogers-McCulloch model.
An implementation of a numerical discretization of the model is provided in the file `TF2D.py`.
The numerical simulation comprises two sinus activation followed by an extra-stimulus which triggers a sustained re-entrant circuits. All the physical coefficients of the model are well defined, outside of $`\nu_2`$, that can vary between $`0.0116`$ and $`0.0124`$.

As solver parameters, it is recommended to use the following configurations:
 - $`N = 128`$, $`M = 64`$,  $`\Delta t=0.01`$
 - $`N = 256`$, $`M = 128`$,  $`\Delta t=0.005`$

For the mathematical formulation, three different time windows can be considered:
 - measurement windows $`(0,450)\ ms`$ used by the device for selecting the impulse characteristics;
 - ATP windows $`(450,525)\ ms`$ where the impulse (maximum duration $`10\ ms`$) is delivered;
 - tracking window $`(600,800)\ ms`$ to verify the effectiveness of the strategy.


## :thought_balloon: Authors 
- Davide Carrara ([@davidecarrara98](https://gitlab.com/davidecarrara98))
- Lorenzo Galvan ([@lorenzogalvan](https://gitlab.com/lorenzogalvan))
- Luca Sosta ([@sostaluca](https://gitlab.com/sostaluca))
