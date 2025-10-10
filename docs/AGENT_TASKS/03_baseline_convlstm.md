# Stage 3 — ConvLSTM / ConvGRU


**Goal:** Add temporal recurrence and compare to S2.

**Agent actions**
1. Implement `stormfusion/models/convlstm.py` (cell + encoder-decoder).
2. Add config `configs/model/convlstm.yaml` and update train script to select model by name.
3. Train on Subset-S with same schedule; compare metrics by lead time.

**Acceptance criteria**
- ConvLSTM achieves better POD at longer leads vs. U-Net MSE on Subset-S.
