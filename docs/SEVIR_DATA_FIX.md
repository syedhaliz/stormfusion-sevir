# SEVIR Data Loader Fix - October 12, 2025

## Problem Summary

The SEVIR dataset has **3 fundamentally different data formats** that the original loader wasn't designed to handle:

1. **VIL (Radar)**: Standard indexed gridded data (384×384×49) ✅
2. **IR069/IR107 (Infrared)**: Indexed gridded data but 192×192 - needs upsampling to 384×384 ⚠️
3. **Lightning (GLM)**: Event-ID-keyed sparse point data - needs grid conversion ❌

The original loader assumed all modalities used `h5[modality][file_index]` access with 384×384 grids, which **fails for IR and lightning**.

---

## Root Cause Analysis

### Diagnostic Approach
Created `scripts/inspect_sevir_files.py` to directly inspect H5 file structure instead of trial-and-error fixes.

### Key Findings from Diagnostic

**VIL Files:**
```python
h5['vil'][index]  # → (384, 384, 49) uint8 [0-255]
```
✅ Works as expected

**IR Files:**
```python
h5['ir069'][index]  # → (192, 192, 49) int16 [-5104, -3663]
h5['ir107'][index]  # → (192, 192, 49) int16 [-4500, -987]
```
⚠️ Resolution mismatch! 192×192 instead of 384×384

**Lightning Files:**
```python
# NO 'lght' key! Uses event_id as key:
h5['R19010510527286']  # → (1172, 5) sparse points
h5['R19010510527663']  # → (0, 5) no lightning

# Format: (N_flashes, 5) where columns are:
# [flash_id, x_coord, y_coord, time_index, energy]
```
❌ Completely different structure - sparse points, not grids!

---

## Solution Implemented

### 1. Rewritten `_load_modality()` Method

**Location:** `stormfusion/data/sevir_multimodal.py:146-195`

```python
def _load_modality(self, event_id, modality):
    """
    Handles 3 different SEVIR data formats:
    - VIL: Indexed gridded (384, 384, 49)
    - IR: Indexed gridded (192, 192, 49) → upsampled to (384, 384, 49)
    - Lightning: Event-ID-keyed sparse → converted to grid (384, 384, 49)
    """
    with h5py.File(info['path'], 'r') as h5:
        if modality == 'lght':
            # Lightning: Use event_id as key
            points = h5[event_id][:]  # (N_flashes, 5)
            data = self._convert_lightning_to_grid(points)

        elif modality in ['ir069', 'ir107']:
            # IR: Load 192×192, upsample to 384×384
            data = h5[modality][info['index']]  # (192, 192, 49)
            data = self._upsample_ir(data)
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)

        else:  # vil
            # VIL: Standard access
            data = h5[modality][info['index']]  # (384, 384, 49)
            data = data / 255.0

    return data
```

### 2. Added Helper Functions

**IR Upsampling** (`_upsample_ir()`):
- Uses scipy.ndimage.zoom with bilinear interpolation
- Upsamples from 192×192 to 384×384
- Preserves temporal dimension (49 frames)

**Lightning Grid Conversion** (`_convert_lightning_to_grid()`):
- Converts sparse point data (N, 5) to grid (384, 384, 49)
- Extracts x, y, time coordinates and energy from point data
- Accumulates lightning energy at grid cells
- Normalizes by max energy to [0, 1] range
- Handles empty lightning (0 flashes) gracefully

---

## Testing

### Test Script
**Location:** `scripts/test_fixed_dataloader.py`

**Run in Colab:**
```python
!cd /content/stormfusion-sevir && python scripts/test_fixed_dataloader.py
```

**Run locally:**
```bash
cd /path/to/stormfusion-sevir
python scripts/test_fixed_dataloader.py
```

### Expected Output
```
✅ Sample loaded successfully!

VIL     : Shape (12, 384, 384) ✅
IR069   : Shape (12, 384, 384) ✅ (upsampled from 192×192)
IR107   : Shape (12, 384, 384) ✅ (upsampled from 192×192)
LGHT    : Shape (12, 384, 384) ✅ (converted from sparse points)
```

---

## Updated Notebooks

All notebooks have been updated with:
1. Google Drive mounting (`drive.mount()`)
2. Repository cloning/updating (`git clone/pull`)
3. Path compatibility (`os.path` instead of `pathlib`)
4. 2019-only data filtering
5. Correct index building

**Updated notebooks:**
- ✅ `02_Data_Verification.ipynb`
- ✅ `03_Test_DataLoader.ipynb`
- ✅ `05_Test_Full_Model.ipynb`
- ✅ `06_Small_Scale_Training.ipynb`
- ✅ `07_Full_Training.ipynb`

---

## Next Steps

### Immediate
1. **Test data loader** in Colab with `03_Test_DataLoader.ipynb`
2. **Verify all modalities load correctly** (VIL, IR, Lightning)
3. **Check shapes** are all (B, T, 384, 384)

### Then
4. Test model components with `04_Test_Model_Components.ipynb`
5. Test full model with `05_Test_Full_Model.ipynb`
6. Run small-scale training with `06_Small_Scale_Training.ipynb`
7. Run full training with `07_Full_Training.ipynb`

---

## Files Modified

1. **`stormfusion/data/sevir_multimodal.py`**
   - Rewritten `_load_modality()` method (lines 146-195)
   - Added `_upsample_ir()` method (lines 197-211)
   - Added `_convert_lightning_to_grid()` method (lines 213-254)

2. **`docs/PROGRESS_UPDATE_OCT12.md`**
   - Added "Critical Data Structure Discovery" section
   - Updated "Current Status" and "Known Issues"
   - Added update log entry

3. **`scripts/test_fixed_dataloader.py`** (new)
   - Test script to validate the fix

4. **`docs/SEVIR_DATA_FIX.md`** (this document)
   - Complete documentation of the fix

---

## Key Takeaways

### What We Learned
1. ✅ **Diagnostic approach > trial-and-error**: Inspecting actual data reveals root cause
2. ✅ **Don't assume uniform structure**: Real datasets have heterogeneous formats
3. ✅ **Read the data format docs**: SEVIR paper mentions different resolutions and formats
4. ✅ **Test incrementally**: Modular notebooks help isolate issues

### Technical Details
- Lightning data is **sparse by nature** - most grid cells have no lightning
- IR satellites have **lower spatial resolution** than radar
- Event IDs serve as **primary keys** in lightning files, not array indices
- SEVIR catalog references **2017-2019** but public bucket only has **2019**

---

## Troubleshooting

**If you still see errors:**

1. **"Event not found in lightning file"**
   - Normal! Not all events have lightning
   - Loader returns zeros (correct behavior)

2. **"All zeros for lightning"**
   - Could be no lightning occurred (valid)
   - Or missing lightning file (check file exists)

3. **"Wrong shape"**
   - Check scipy is installed: `pip install scipy`
   - Needed for IR upsampling

4. **"File not found"**
   - Verify data downloaded to Drive
   - Check paths in notebook match Drive structure

---

**Fix implemented:** October 12, 2025
**Status:** ✅ Complete and ready for testing
**Next:** Run `03_Test_DataLoader.ipynb` in Colab
