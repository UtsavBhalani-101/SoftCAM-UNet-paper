For the training data, we use selected cases from [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)
First we need to extract the group the DICOM slices from their z-position and StudyInstanceUID, we provide the [reoragnized label list](https://drive.google.com/file/d/1QGcGmIYAzSCMC7QJbJnUXTdqWVS92lQT/view?usp=sharing)

To extract the training data from the RSNA dataset, execute the following:

```
python create_rsna_dataset.py --csv_path <the path you put the above label list > --dcm-dir <the dir of RSNA DICOM images> --save_dir <the image dir you want to create> 
```

, and then use [dcm_to_png_3ch](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/utils/transfer_to_png.py) to transfer the DICOM files into our PNG formats.


For the test data, we test on the [BHSD dataset](https://github.com/White65534/BHSD). For the preprocessing, use [nii_to_png_3ch](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/utils/transfer_to_png.py) to transfer the NIFTI files into our PNG formats. Then iteratively use [demo_code.py](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/demo_code.py) to visualize the generated heatmaps.




